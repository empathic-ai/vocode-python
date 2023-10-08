import abc
import os
from functools import partial
import logging
from typing import List, Optional, Dict
from fastapi import APIRouter, Form, Request, Response, HTTPException
from pydantic import BaseModel, Field
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.models.events import RecordingEvent
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.synthesizer.factory import SynthesizerFactory
from vocode.streaming.telephony.client.base_telephony_client import BaseTelephonyClient
from vocode.streaming.telephony.client.twilio_client import TwilioClient
from vocode.streaming.telephony.client.vonage_client import VonageClient
from vocode.streaming.telephony.config_manager.base_config_manager import (
    BaseConfigManager,
)
from vocode import getenv
from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLING_RATE,
    VONAGE_AUDIO_ENCODING,
    VONAGE_SAMPLING_RATE,
)

from vocode.streaming.telephony.server.router.calls import CallsRouter
from vocode.streaming.models.telephony import (
    TwilioCallConfig,
    TwilioConfig,
    VonageCallConfig,
    VonageConfig,
)

from vocode.streaming.telephony.templater import Templater
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.transcriber.factory import TranscriberFactory
from vocode.streaming.utils import create_conversation_id
from vocode.streaming.utils.events_manager import EventsManager
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from vocode.streaming.transcriber.base_transcriber import (
    Transcription
)
from vocode.streaming.telephony.call_manager import CallManager
from vocode.streaming.agent.base_agent import (
    TranscriptionAgentInput
)
from vocode.streaming.utils.worker import (
    InterruptibleEvent
)

class AbstractInboundCallConfig(BaseModel, abc.ABC):
    url: str
    agent_config: AgentConfig
    transcriber_config: Optional[TranscriberConfig] = None
    synthesizer_config: Optional[SynthesizerConfig] = None


class TwilioInboundCallConfig(AbstractInboundCallConfig):
    twilio_config: TwilioConfig


class VonageInboundCallConfig(AbstractInboundCallConfig):
    vonage_config: VonageConfig


class VonageAnswerRequest(BaseModel):
    to: str
    from_: str = Field(..., alias="from")
    uuid: str


class TelephonyServer:
    call_manager: CallManager = CallManager()

    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        inbound_call_configs: List[AbstractInboundCallConfig] = [],
        transcriber_factory: TranscriberFactory = TranscriberFactory(),
        agent_factory: AgentFactory = AgentFactory(),
        synthesizer_factory: SynthesizerFactory = SynthesizerFactory(),
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.router = APIRouter()
        self.config_manager = config_manager
        self.templater = Templater()
        self.events_manager = events_manager
        self.router.include_router(
            CallsRouter(
                base_url=base_url,
                config_manager=self.config_manager,
                transcriber_factory=transcriber_factory,
                agent_factory=agent_factory,
                synthesizer_factory=synthesizer_factory,
                events_manager=self.events_manager,
                call_manager=self.call_manager,
                logger=self.logger,
            ).get_router()
        )
        for config in inbound_call_configs:
            self.router.add_api_route(
                config.url,
                self.create_inbound_route(inbound_call_config=config),
                methods=["POST"],
            )

        # twilio SMS endpoint
        self.router.add_api_route("/sms", self.handle_sms, methods=["GET", "POST"])

        # vonage requires an events endpoint
        self.router.add_api_route("/events", self.events, methods=["GET", "POST"])
        self.logger.info(f"Set up events endpoint at https://{self.base_url}/events")

        self.router.add_api_route("/recordings/{conversation_id}", self.recordings, methods=["GET", "POST"])
        self.logger.info(f"Set up recordings endpoint at https://{self.base_url}/recordings/{{conversation_id}}")

    async def validate_twilio_request(self, request: Request) -> bool:
        VALIDATOR = RequestValidator(getenv("TWILIO_AUTH_TOKEN"))
        signature = request.headers.get("X-Twilio-Signature")
        url = str(request.url)
        params = await request.form()
        return VALIDATOR.validate(url, params, signature)

    async def handle_sms(self, request: Request):
        if not await self.validate_twilio_request(request):
            raise HTTPException(status_code=400, detail="Invalid Twilio request")

        # Extract the message body from the incoming POST request
        form_data = await request.form()
        incoming_msg = form_data.get("Body")
        phone_number = form_data.get("From")

        # Process the incoming message (this step is up to you)
        # Here's a simple example that echoes back the received SMS
        reply_msg = f"You said: {incoming_msg}"
        
        if phone_number in self.call_manager.agents_by_number:
            self.logger.info(f"Phone number present in call manager!")
            agent = self.call_manager.agents_by_number[phone_number]
            conversation = agent.conversation_state_manager.conversation
            conversation_id = conversation.id
            self.call_manager.agents_by_number[phone_number].input_queue.put_nowait(
                InterruptibleEvent(
                    payload=TranscriptionAgentInput(
                        transcription=Transcription(
                            message=incoming_msg,
                            confidence=1.0,
                            is_final=False,
                        ),
                        conversation_id=conversation_id,
                        vonage_uuid=getattr(conversation, "vonage_uuid", None),
                        twilio_sid=getattr(conversation, "twilio_sid", None),
                    ),
                    is_interruptible=False
                )
            )
        
        self.logger.info(f"Responding to message '{incoming_msg}' from {phone_number} with '{reply_msg}'!")
        #self.logger.info(f"Responding to form '{form_data}'!")

        # Create a TwiML response
        response = MessagingResponse()
        response.message(reply_msg)

        return Response(content=str(response), media_type="text/html")

    def events(self, request: Request):
        return Response()

    async def recordings(self, request: Request, conversation_id: str):
        recording_url = (await request.json())["recording_url"]
        if self.events_manager is not None and recording_url is not None:
            self.events_manager.publish_event(RecordingEvent(recording_url=recording_url, conversation_id=conversation_id))
        return Response()

    def create_inbound_route(
        self,
        inbound_call_config: AbstractInboundCallConfig,
    ):
        async def twilio_route(
            twilio_config: TwilioConfig,
            twilio_sid: str = Form(alias="CallSid"),
            twilio_from: str = Form(alias="From"),
            twilio_to: str = Form(alias="To"),
        ) -> Response:
            call_config = TwilioCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or TwilioCallConfig.default_transcriber_config(),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or TwilioCallConfig.default_synthesizer_config(),
                twilio_config=twilio_config,
                twilio_sid=twilio_sid,
                from_phone=twilio_from,
                to_phone=twilio_to,
            )

            conversation_id = create_conversation_id()
            await self.config_manager.save_config(conversation_id, call_config)
            return self.templater.get_connection_twiml(
                base_url=self.base_url, call_id=conversation_id
            )

        async def vonage_route(
            vonage_config: VonageConfig, vonage_answer_request: VonageAnswerRequest
        ):
            call_config = VonageCallConfig(
                transcriber_config=inbound_call_config.transcriber_config
                or VonageCallConfig.default_transcriber_config(),
                agent_config=inbound_call_config.agent_config,
                synthesizer_config=inbound_call_config.synthesizer_config
                or VonageCallConfig.default_synthesizer_config(),
                vonage_config=vonage_config,
                vonage_uuid=vonage_answer_request.uuid,
                to_phone=vonage_answer_request.from_,
                from_phone=vonage_answer_request.to,
            )
            conversation_id = create_conversation_id()
            await self.config_manager.save_config(conversation_id, call_config)
            return VonageClient.create_call_ncco(
                base_url=self.base_url, conversation_id=conversation_id, record=vonage_config.record
            )

        if isinstance(inbound_call_config, TwilioInboundCallConfig):
            self.logger.info(
                f"Set up inbound call TwiML at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(twilio_route, inbound_call_config.twilio_config)
        elif isinstance(inbound_call_config, VonageInboundCallConfig):
            self.logger.info(
                f"Set up inbound call NCCO at https://{self.base_url}{inbound_call_config.url}"
            )
            return partial(vonage_route, inbound_call_config.vonage_config)
        else:
            raise ValueError(
                f"Unknown inbound call config type {type(inbound_call_config)}"
            )

    async def end_outbound_call(self, conversation_id: str):
        # TODO validation via twilio_client
        call_config = await self.config_manager.get_config(conversation_id)
        if not call_config:
            raise ValueError(f"Could not find call config for {conversation_id}")
        telephony_client: BaseTelephonyClient
        if isinstance(call_config, TwilioCallConfig):
            telephony_client = TwilioClient(
                base_url=self.base_url, twilio_config=call_config.twilio_config
            )
            await telephony_client.end_call(call_config.twilio_sid)
        elif isinstance(call_config, VonageCallConfig):
            telephony_client = VonageClient(
                base_url=self.base_url, vonage_config=call_config.vonage_config
            )
            await telephony_client.end_call(call_config.vonage_uuid)
        return {"id": conversation_id}

    def get_router(self) -> APIRouter:
        return self.router
