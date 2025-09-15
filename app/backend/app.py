import dataclasses
import io
import json
import logging
import mimetypes
import math  # added for cosine similarity
import os
import time
import traceback
from collections.abc import AsyncGenerator, Awaitable
from pathlib import Path
from typing import Any, Callable, Union, cast


from azure.cognitiveservices.speech import (
    ResultReason,
    SpeechConfig,
    SpeechSynthesisOutputFormat,
    SpeechSynthesisResult,
    SpeechSynthesizer,
)
from azure.identity.aio import (
    AzureDeveloperCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.resources import Resource  # Add this import
from quart import (
    Blueprint,
    Quart,
    abort,
    current_app,
    jsonify,
    make_response,
    request,
    send_file,
    send_from_directory,
)
from quart_cors import cors

from dotenv import load_dotenv

from approaches.approach import Approach
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.promptmanager import PromptyManager
from approaches.retrievethenread import RetrieveThenReadApproach
from chat_history.cosmosdb import chat_history_cosmosdb_bp
from config import (
    CONFIG_AGENT_CLIENT,
    CONFIG_AGENTIC_RETRIEVAL_ENABLED,
    CONFIG_ASK_APPROACH,
    CONFIG_AUTH_CLIENT,
    CONFIG_CHAT_APPROACH,
    CONFIG_CHAT_HISTORY_BROWSER_ENABLED,
    CONFIG_CHAT_HISTORY_COSMOS_ENABLED,
    CONFIG_CREDENTIAL,
    CONFIG_DEFAULT_REASONING_EFFORT,
    CONFIG_GLOBAL_BLOB_MANAGER,
    CONFIG_INGESTER,
    CONFIG_LANGUAGE_PICKER_ENABLED,
    CONFIG_MULTIMODAL_ENABLED,
    CONFIG_OPENAI_CLIENT,
    CONFIG_QUERY_REWRITING_ENABLED,
    CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS,
    CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS,
    CONFIG_RAG_SEND_IMAGE_SOURCES,
    CONFIG_RAG_SEND_TEXT_SOURCES,
    CONFIG_REASONING_EFFORT_ENABLED,
    CONFIG_SEARCH_CLIENT,
    CONFIG_SEMANTIC_RANKER_DEPLOYED,
    CONFIG_SPEECH_INPUT_ENABLED,
    CONFIG_SPEECH_OUTPUT_AZURE_ENABLED,
    CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED,
    CONFIG_SPEECH_SERVICE_ID,
    CONFIG_SPEECH_SERVICE_LOCATION,
    CONFIG_SPEECH_SERVICE_TOKEN,
    CONFIG_SPEECH_SERVICE_VOICE,
    CONFIG_STREAMING_ENABLED,
    CONFIG_USER_BLOB_MANAGER,
    CONFIG_USER_UPLOAD_ENABLED,
    CONFIG_VECTOR_SEARCH_ENABLED,
)
from core.authentication import AuthenticationHelper
from core.sessionhelper import create_session_id
from decorators import authenticated, authenticated_path
from error import error_dict, error_response
from prepdocs import (
    OpenAIHost,
    clean_key_if_exists,
    setup_embeddings_service,
    setup_file_processors,
    setup_image_embeddings_service,
    setup_openai_client,
    setup_search_info,
)
from prepdocslib.blobmanager import AdlsBlobManager, BlobManager
from prepdocslib.embeddings import ImageEmbeddings
from prepdocslib.filestrategy import UploadUserFileStrategy
from prepdocslib.listfilestrategy import File


import asyncio
from typing import Any, Awaitable

# Telemetry init guard to avoid double-instrumentation across forks/workers
_TELEMETRY_INITIALIZED = False
_TELEMETRY_LOCK = asyncio.Lock()

async def _safe_wait(coro: Awaitable[Any], timeout: float, name: str = "<operation>") -> Any:
    """
    Await 'coro' but bound it to 'timeout' seconds. If it times out or raises,
    log and return None (caller must handle None).
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logging.warning("Timed out after %ss while running %s", timeout, name)
        return None
    except Exception as e:
        logging.exception("Exception during %s: %s", name, e)
        return None

async def _init_telemetry_once(app: Quart):
    """Initialize telemetry with proper locking to prevent race conditions"""
    global _TELEMETRY_INITIALIZED
    
    async with _TELEMETRY_LOCK:
        if _TELEMETRY_INITIALIZED:
            app.logger.debug("Telemetry already initialized, skipping")
            return
            
        if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
            app.logger.info("APPLICATIONINSIGHTS_CONNECTION_STRING is set, enabling Azure Monitor")
            try:
                # Configure with latest 2025 recommendations
                configure_azure_monitor(
                    instrumentation_options={
                        "django": {"enabled": False},
                        "psycopg2": {"enabled": False},
                        "fastapi": {"enabled": False},
                        "flask": {"enabled": False},  # Disable Flask instrumentation for Quart
                        "urllib": {"enabled": True},   # Enable urllib instrumentation
                        "urllib3": {"enabled": True},  # Enable urllib3 instrumentation
                    },
                    # Create proper Resource object with attributes
                    resource=Resource.create({
                        "service.name": os.environ.get("AZURE_APP_SERVICE_NAME", "azure-search-openai-demo"),
                        "service.version": os.environ.get("APP_VERSION", "1.0.0"),
                        "deployment.environment": os.environ.get("ENVIRONMENT", "production" if os.environ.get("WEBSITE_HOSTNAME") else "development"),
                    })
                )
                
                # Instrument HTTP clients
                AioHttpClientInstrumentor().instrument()
                HTTPXClientInstrumentor().instrument()
                OpenAIInstrumentor().instrument()
                
                # Add ASGI middleware for request tracking
                app.asgi_app = OpenTelemetryMiddleware(app.asgi_app)  # type: ignore[assignment]
                
                app.logger.info("Azure Monitor configured successfully")
                _TELEMETRY_INITIALIZED = True
            except Exception as e:
                app.logger.exception("Failed to configure Azure Monitor (continuing without telemetry): %s", e)

bp = Blueprint("routes", __name__, static_folder="static")

# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


@bp.route("/")
async def index():
    return await bp.send_static_file("index.html")


# Empty page is recommended for login redirect to work.
# See https://github.com/AzureAD/microsoft-authentication-library-for-js/blob/dev/lib/msal-browser/docs/initialization.md#redirecturi-considerations for more information
@bp.route("/redirect")
async def redirect():
    return ""


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory(Path(__file__).resolve().parent / "static" / "assets", path)


@bp.route("/content/<path>")
@authenticated_path
async def content_file(path: str, auth_claims: dict[str, Any]):
    """
    Serve content files from blob storage from within the app to keep the example self-contained.
    *** NOTE *** if you are using app services authentication, this route will return unauthorized to all users that are not logged in
    if AZURE_ENFORCE_ACCESS_CONTROL is not set or false, logged in users can access all files regardless of access control
    if AZURE_ENFORCE_ACCESS_CONTROL is set to true, logged in users can only access files they have access to
    This is also slow and memory hungry.
    """
    # Remove page number from path, filename-1.txt -> filename.txt
    # This shouldn't typically be necessary as browsers don't send hash fragments to servers
    if path.find("#page=") > 0:
        path_parts = path.rsplit("#page=", 1)
        path = path_parts[0]
    current_app.logger.info("Opening file %s", path)
    blob_manager: BlobManager = current_app.config[CONFIG_GLOBAL_BLOB_MANAGER]

    # Get bytes and properties from the blob manager
    result = await blob_manager.download_blob(path)

    if result is None:
        current_app.logger.info("Path not found in general Blob container: %s", path)
        if current_app.config[CONFIG_USER_UPLOAD_ENABLED]:
            user_oid = auth_claims["oid"]
            user_blob_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
            result = await user_blob_manager.download_blob(path, user_oid=user_oid)
            if result is None:
                current_app.logger.exception("Path not found in DataLake: %s", path)

    if not result:
        abort(404)

    content, properties = result

    if not properties or "content_settings" not in properties:
        abort(404)

    mime_type = properties["content_settings"]["content_type"]
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    # Create a BytesIO object from the bytes
    blob_file = io.BytesIO(content)
    return await send_file(blob_file, mimetype=mime_type, as_attachment=False, attachment_filename=path)


def _validate_messages_payload(request_json: dict) -> None:
	"""
	Validate the shape of the incoming request JSON used by /ask and /chat endpoints.
	Raises ValueError with a descriptive message when the payload is malformed.
	"""
	if "messages" not in request_json:
		raise ValueError("Missing 'messages' array in request JSON.")
	messages = request_json["messages"]
	if not isinstance(messages, list):
		raise ValueError("'messages' must be a list.")
	if len(messages) == 0:
		raise ValueError("'messages' must contain at least one message.")
	for idx, m in enumerate(messages):
		if not isinstance(m, dict):
			raise ValueError(f"Message at index {idx} is not an object.")
		if "role" not in m:
			raise ValueError(f"Message at index {idx} is missing required property 'role'.")
		# content may be nested in some clients; check conservative condition
		if "content" not in m and "text" not in m and "message" not in m:
			# don't be too strict, but warn / fail if nothing that looks like content exists
			raise ValueError(f"Message at index {idx} is missing content (no 'content'/'text'/'message').")


@bp.route("/ask", methods=["POST"])
@authenticated
async def ask(auth_claims: dict[str, Any]):
    start_time = time.time()  # Start timing
    user_id = auth_claims.get("oid", "anonymous")
    
    # Track request
    metrics_store.add_request("ask", user_id)
    request_counter.add(1, {"endpoint": "ask", "user": user_id})
    
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    # Log raw JSON only in debug mode to avoid leaking user content in production
    if os.environ.get("APP_LOG_LEVEL", "").upper() == "DEBUG" or os.environ.get("SHOW_ERRORS", "").lower() == "true":
        current_app.logger.debug("Incoming /ask JSON: %s", request_json)

    try:
        _validate_messages_payload(request_json)
        
        # Track query
        if request_json.get("messages"):
            last_message = request_json["messages"][-1]
            if last_message.get("content"):
                metrics_store.add_query(str(last_message["content"]))
        
        # Track index access
        if index_name := os.environ.get("AZURE_SEARCH_INDEX"):
            metrics_store.add_index_access(index_name)

        context = request_json.get("context", {})
        context["auth_claims"] = auth_claims
        
        with tracer.start_as_current_span("ask_request") as span:
            span.set_attribute("user.id", user_id)
            span.set_attribute("endpoint", "ask")
            span.set_attribute("model", os.environ.get("AZURE_OPENAI_CHATGPT_MODEL", "unknown"))
            
            approach: Approach = cast(Approach, current_app.config[CONFIG_ASK_APPROACH])
            r = await approach.run(
                request_json["messages"], context=context, session_state=request_json.get("session_state")
            )
            
            # Extract token usage from response if available
            if isinstance(r, dict):
                # Check for usage information in various formats
                usage = r.get("usage") or r.get("token_usage") or {}
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    metrics_store.add_tokens(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        user_id=user_id
                    )
                    token_counter.add(total_tokens, {"endpoint": "ask", "user": user_id})
                    span.set_attribute("tokens.prompt", prompt_tokens)
                    span.set_attribute("tokens.completion", completion_tokens)
                    span.set_attribute("tokens.total", total_tokens)
        
        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        metrics_store.add_latency(latency_ms)
        latency_histogram.record(latency_ms, {"endpoint": "ask", "status": "success"})
        
        return jsonify(r)
    except Exception as error:
        # Track failed request
        request_counter.add(1, {"endpoint": "ask", "status": "error", "user": user_id})
        
        latency_ms = (time.time() - start_time) * 1000
        latency_histogram.record(latency_ms, {"endpoint": "ask", "status": "error"})
        
        current_app.logger.error("Error in /ask endpoint: %s", str(error))
        return error_response(error, "/ask")


@bp.route("/chat", methods=["POST"])
@authenticated
async def chat(auth_claims: dict[str, Any]):
    start_time = time.time()
    user_id = auth_claims.get("oid", "anonymous")
    
    # Track request
    metrics_store.add_request("chat", user_id)
    request_counter.add(1, {"endpoint": "chat", "user": user_id})
    
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    # Debug-only raw request log
    if os.environ.get("APP_LOG_LEVEL", "").upper() == "DEBUG" or os.environ.get("SHOW_ERRORS", "").lower() == "true":
        current_app.logger.debug("Incoming /chat JSON: %s", request_json)

    try:
        _validate_messages_payload(request_json)
        
        # Track query
        if request_json.get("messages"):
            last_message = request_json["messages"][-1]
            if last_message.get("content"):
                metrics_store.add_query(str(last_message["content"]))
        
        # Track index access
        if index_name := os.environ.get("AZURE_SEARCH_INDEX"):
            metrics_store.add_index_access(index_name)

        context = request_json.get("context", {})
        context["auth_claims"] = auth_claims

        with tracer.start_as_current_span("chat_request") as span:
            span.set_attribute("user.id", user_id)
            span.set_attribute("endpoint", "chat")
            span.set_attribute("model", os.environ.get("AZURE_OPENAI_CHATGPT_MODEL", "unknown"))
            
            approach: Approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])

            # If session state is provided, persists the session state,
            # else creates a new session_id depending on the chat history options enabled.
            session_state = request_json.get("session_state")
            if session_state is None:
                session_state = create_session_id(
                    current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
                    current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
                )
            
            result = await approach.run(
                request_json["messages"],
                context=context,
                session_state=session_state,
            )
            
            # Extract token usage
            if isinstance(result, dict):
                usage = result.get("usage") or result.get("token_usage") or {}
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    metrics_store.add_tokens(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        user_id=user_id
                    )
                    token_counter.add(total_tokens, {"endpoint": "chat", "user": user_id})
                    span.set_attribute("tokens.prompt", prompt_tokens)
                    span.set_attribute("tokens.completion", completion_tokens)
                    span.set_attribute("tokens.total", total_tokens)
        
        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        metrics_store.add_latency(latency_ms)
        latency_histogram.record(latency_ms, {"endpoint": "chat", "status": "success"})
        
        return jsonify(result)
    except Exception as error:
        request_counter.add(1, {"endpoint": "chat", "status": "error", "user": user_id})
        
        latency_ms = (time.time() - start_time) * 1000
        latency_histogram.record(latency_ms, {"endpoint": "chat", "status": "error"})
        
        current_app.logger.error("Error in /chat endpoint: %s", str(error))
        return error_response(error, "/chat")


@bp.route("/chat/stream", methods=["POST"])
@authenticated
async def chat_stream(auth_claims: dict[str, Any]):
    # NEW: start timing + metrics for streaming (was missing)
    start_time = time.time()
    user_id = auth_claims.get("oid", "anonymous")
    metrics_store.add_request("chat_stream", user_id)
    request_counter.add(1, {"endpoint": "chat_stream", "user": user_id})
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    if os.environ.get("APP_LOG_LEVEL", "").upper() == "DEBUG" or os.environ.get("SHOW_ERRORS", "").lower() == "true":
        current_app.logger.debug("Incoming /chat/stream JSON: %s", request_json)
    try:
        _validate_messages_payload(request_json)

        # --- NEW: track query + index access for streaming path ---
        if request_json.get("messages"):
            last_message = request_json["messages"][-1]
            if last_message.get("content"):
                metrics_store.add_query(str(last_message["content"]))
        if index_name := os.environ.get("AZURE_SEARCH_INDEX"):
            metrics_store.add_index_access(index_name)
        # ----------------------------------------------------------

        context = request_json.get("context", {})
        context["auth_claims"] = auth_claims
        approach: Approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])
        session_state = request_json.get("session_state")
        if session_state is None:
            session_state = create_session_id(
                current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
                current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
            )
        # Start span manually to close when stream finishes
        span = tracer.start_span("chat_stream_request")
        span.set_attribute("user.id", user_id)
        span.set_attribute("endpoint", "chat_stream")
        span.set_attribute("model", os.environ.get("AZURE_OPENAI_CHATGPT_MODEL", "unknown"))
        raw_stream = await approach.run_stream(
            request_json["messages"],
            context=context,
            session_state=session_state,
        )
        # Wrapper generator to intercept usage + finalize metrics
        async def instrument_stream():
            try:
                async for event in raw_stream:
                    # If usage appears (added in approach), update metrics
                    if "usage" in event:
                        usage = event["usage"] or {}
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                        if total_tokens:
                            metrics_store.add_tokens(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                user_id=user_id,
                            )
                            token_counter.add(total_tokens, {"endpoint": "chat_stream", "user": user_id})
                            span.set_attribute("tokens.prompt", prompt_tokens)
                            span.set_attribute("tokens.completion", completion_tokens)
                            span.set_attribute("tokens.total", total_tokens)
                    yield event
            finally:
                latency_ms = (time.time() - start_time) * 1000
                metrics_store.add_latency(latency_ms)
                latency_histogram.record(latency_ms, {"endpoint": "chat_stream", "status": "success"})
                span.end()
        response = await make_response(format_as_ndjson(instrument_stream()))
        response.timeout = None  # type: ignore
        response.mimetype = "application/json-lines"
        return response
    except Exception as error:
        latency_ms = (time.time() - start_time) * 1000
        latency_histogram.record(latency_ms, {"endpoint": "chat_stream", "status": "error"})
        current_app.logger.error("Error in /chat/stream endpoint: %s", str(error))
        return error_response(error, "/chat")


# Send MSAL.js settings to the client UI
@bp.route("/auth_setup", methods=["GET"])
def auth_setup():
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    return jsonify(auth_helper.get_auth_setup_for_client())


@bp.route("/config", methods=["GET"])
def config():
    return jsonify(
        {
            "showMultimodalOptions": current_app.config[CONFIG_MULTIMODAL_ENABLED],
            "showSemanticRankerOption": current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED],
            "showQueryRewritingOption": current_app.config[CONFIG_QUERY_REWRITING_ENABLED],
            "showReasoningEffortOption": current_app.config[CONFIG_REASONING_EFFORT_ENABLED],
            "streamingEnabled": current_app.config[CONFIG_STREAMING_ENABLED],
            "defaultReasoningEffort": current_app.config[CONFIG_DEFAULT_REASONING_EFFORT],
            "showVectorOption": current_app.config[CONFIG_VECTOR_SEARCH_ENABLED],
            "showUserUpload": current_app.config[CONFIG_USER_UPLOAD_ENABLED],
            "showLanguagePicker": current_app.config[CONFIG_LANGUAGE_PICKER_ENABLED],
            "showSpeechInput": current_app.config[CONFIG_SPEECH_INPUT_ENABLED],
            "showSpeechOutputBrowser": current_app.config[CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED],
            "showSpeechOutputAzure": current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED],
            "showChatHistoryBrowser": current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED],
            "showChatHistoryCosmos": current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED],
            "showAgenticRetrievalOption": current_app.config[CONFIG_AGENTIC_RETRIEVAL_ENABLED],
            "ragSearchTextEmbeddings": current_app.config[CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS],
            "ragSearchImageEmbeddings": current_app.config[CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS],
            "ragSendTextSources": current_app.config[CONFIG_RAG_SEND_TEXT_SOURCES],
            "ragSendImageSources": current_app.config[CONFIG_RAG_SEND_IMAGE_SOURCES],
        }
    )


@bp.route("/speech", methods=["POST"])
async def speech():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415

    speech_token = current_app.config.get(CONFIG_SPEECH_SERVICE_TOKEN)
    if speech_token is None or speech_token.expires_on < time.time() + 60:
        speech_token = await current_app.config[CONFIG_CREDENTIAL].get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        current_app.config[CONFIG_SPEECH_SERVICE_TOKEN] = speech_token

    request_json = await request.get_json()
    text = request_json.get("text", "")
    if text == "":
        return jsonify({"error": "text is required"}), 400

    try:
        # Construct a token as described in documentation:
        # https://learn.microsoft.com/azure/ai-services/speech-service/how-to-configure-azure-ad-auth?pivots=programming-language-python
        auth_token = (
            "aad#"
            + current_app.config[CONFIG_SPEECH_SERVICE_ID]
            + "#"
            + current_app.config[CONFIG_SPEECH_SERVICE_TOKEN].token
        )
        speech_config = SpeechConfig(auth_token=auth_token, region=current_app.config[CONFIG_SPEECH_SERVICE_LOCATION])
        speech_config.speech_synthesis_voice_name = current_app.config[CONFIG_SPEECH_SERVICE_VOICE]
        speech_config.speech_synthesis_output_format = SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result: SpeechSynthesisResult = synthesizer.speak_text_async(text).get()
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return result.audio_data, 200, {"Content-Type": "audio/mp3"}
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            current_app.logger.error(
                "Speech synthesis canceled: %s %s", cancellation_details.reason, cancellation_details.error_details
            )
            raise Exception("Speech synthesis canceled. Check logs for details.")
        else:
            current_app.logger.error("Unexpected result reason: %s", result.reason)
            raise Exception("Speech synthesis failed. Check logs for details.")
    except Exception as e:
        current_app.logger.exception("Exception in /speech")
        return jsonify({"error": str(e)}), 500


@bp.route("/speech/token", methods=["POST"])
async def speech_token():
    azure_credential = current_app.config[CONFIG_CREDENTIAL]
    speech_token = await azure_credential.get_token("https://cognitiveservices.azure.com/.default")
    current_app.config[CONFIG_SPEECH_SERVICE_TOKEN] = speech_token.token

    # To get around an HTTP2 framing issue with bearer token auth that only impacts development
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((f"{current_app.config[CONFIG_SPEECH_SERVICE_ID]}.tts.speech.microsoft.com", 443))
    s.close()

    return jsonify(
        {
            "token": speech_token.token,
            "region": current_app.config[CONFIG_SPEECH_SERVICE_LOCATION],
            "voice": current_app.config[CONFIG_SPEECH_SERVICE_VOICE],
        }
    )


@bp.post("/upload")
@authenticated
async def upload(auth_claims: dict[str, Any]):
    request_files = await request.files
    if "file" not in request_files:
        return jsonify({"message": "No file part in the request", "status": "failed"}), 400

    try:
        user_oid = auth_claims["oid"]
        file = request_files.getlist("file")[0]
        adls_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
        file_url = await adls_manager.upload_blob(file, file.filename, user_oid)
        ingester: UploadUserFileStrategy = current_app.config[CONFIG_INGESTER]
        await ingester.add_file(File(content=file, url=file_url, acls={"oids": [user_oid]}), user_oid=user_oid)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as error:
        current_app.logger.error("Error uploading file: %s", error)
        return jsonify({"message": "Error uploading file, check server logs for details.", "status": "failed"}), 500


@bp.post("/delete_uploaded")
@authenticated
async def delete_uploaded(auth_claims: dict[str, Any]):
    request_json = await request.get_json()
    filename = request_json.get("filename")
    user_oid = auth_claims["oid"]
    try:
        adls_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
        await adls_manager.remove_blob(filename, user_oid)
        ingester: UploadUserFileStrategy = current_app.config[CONFIG_INGESTER]
        await ingester.remove_file(filename, user_oid)
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as error:
        current_app.logger.error("Error deleting file %s: %s", filename, error)
        return jsonify({"message": "Error deleting file, check server logs for details.", "status": "failed"}), 500


@bp.get("/list_uploaded")
@authenticated
async def list_uploaded(auth_claims: dict[str, Any]):
    """Lists the uploaded documents for the current user.
    Only returns files directly in the user's directory, not in subdirectories.
    Excludes image files and the images directory."""
    user_oid = auth_claims["oid"]
    adls_manager: AdlsBlobManager = current_app.config[CONFIG_USER_BLOB_MANAGER]
    files = await adls_manager.list_blobs(user_oid)
    return jsonify(files), 200


@bp.before_app_serving
async def setup_clients():
    """
    Fast startup path: Set up credential synchronously first, then spawn background task for heavy initialization.
    """
    # --- Set up credential synchronously FIRST to avoid race conditions ---
    RUNNING_ON_AZURE = os.environ.get("WEBSITE_HOSTNAME") is not None or os.environ.get("RUNNING_IN_PRODUCTION") is not None
    AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
    
    if RUNNING_ON_AZURE:
        current_app.logger.info("Setting up Azure credential using ManagedIdentityCredential")
        if AZURE_CLIENT_ID := os.environ.get("AZURE_CLIENT_ID"):
            current_app.logger.info("Using user-assigned managed identity with client_id %s", AZURE_CLIENT_ID)
            azure_credential = ManagedIdentityCredential(client_id=AZURE_CLIENT_ID)
        else:
            azure_credential = ManagedIdentityCredential()
    elif AZURE_TENANT_ID:
        current_app.logger.info("Setting up AzureDeveloperCliCredential with tenant_id %s", AZURE_TENANT_ID)
        azure_credential = AzureDeveloperCliCredential(tenant_id=AZURE_TENANT_ID, process_timeout=60)
    else:
        current_app.logger.info("Setting up AzureDeveloperCliCredential for home tenant")
        azure_credential = AzureDeveloperCliCredential(process_timeout=60)

    # Set both keys that might be needed by other modules
    current_app.config[CONFIG_CREDENTIAL] = azure_credential
    current_app.config["azure_credential"] = azure_credential  # For backward compatibility
    
    # Mark not ready until background init completes
    current_app.config["INIT_READY"] = False

    async def _background_init():
        try:
            current_app.logger.info("Background initialization started")
            
            # Credential is already set up above, just get it
            azure_credential = current_app.config[CONFIG_CREDENTIAL]
            azure_ai_token_provider = get_bearer_token_provider(azure_credential, "https://cognitiveservices.azure.com/.default")

            # --- Get all the environment variables we need ---
            # FIX: Use parentheses not square brackets for os.environ.get()
            AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT")
            AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER")
            AZURE_IMAGESTORAGE_CONTAINER = os.environ.get("AZURE_IMAGESTORAGE_CONTAINER")
            AZURE_USERSTORAGE_ACCOUNT = os.environ.get("AZURE_USERSTORAGE_ACCOUNT")
            AZURE_USERSTORAGE_CONTAINER = os.environ.get("AZURE_USERSTORAGE_CONTAINER")

            AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE", "")
            AZURE_SEARCH_ENDPOINT = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net" if AZURE_SEARCH_SERVICE else ""
            AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "")
            AZURE_SEARCH_AGENT = os.environ.get("AZURE_SEARCH_AGENT", "")
            
            # OpenAI configuration - MUST be declared before use
            OPENAI_HOST = OpenAIHost(os.environ.get("OPENAI_HOST", "azure"))
            
            # Used with Azure OpenAI deployments
            AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
            AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
                os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
                if OPENAI_HOST in [OpenAIHost.AZURE, OpenAIHost.AZURE_CUSTOM]
                else None
            )
            AZURE_OPENAI_EMB_DEPLOYMENT = (
                os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT") 
                if OPENAI_HOST in [OpenAIHost.AZURE, OpenAIHost.AZURE_CUSTOM] 
                else None
            )

            # Search agent model/deployment used for retrieval agents (may be different from chat model)
            AZURE_OPENAI_SEARCHAGENT_MODEL = os.environ.get("AZURE_OPENAI_SEARCHAGENT_MODEL")
            AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT")

            # -------- FIX: only use explicit custom URL, do NOT fall back to AZURE_OPENAI_ENDPOINT --------
            raw_custom = os.environ.get("AZURE_OPENAI_CUSTOM_URL")
            if raw_custom:
                # Sanitize URL: remove trailing slash
                trimmed = raw_custom.rstrip("/")
                # Also remove /openai suffix if present to avoid double segments
                if trimmed.endswith("/openai"):
                    current_app.logger.warning("AZURE_OPENAI_CUSTOM_URL ends with /openai; trimming to avoid double segment")
                    trimmed = trimmed[: -len("/openai")]
                AZURE_OPENAI_CUSTOM_URL = trimmed
                current_app.logger.info("Using explicit AZURE_OPENAI_CUSTOM_URL=%s", AZURE_OPENAI_CUSTOM_URL)
            else:
                # No explicit custom URL: let helper compose standard https://{service}.openai.azure.com
                AZURE_OPENAI_CUSTOM_URL = None
                if os.environ.get("AZURE_OPENAI_ENDPOINT"):
                    current_app.logger.info(
                        "Ignoring AZURE_OPENAI_ENDPOINT for custom URL (using service name '%s' instead). "
                        "Set AZURE_OPENAI_CUSTOM_URL to override explicitly.", AZURE_OPENAI_SERVICE
                    )
            # ----------------------------------------------------------------------------------------------

            AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
            AZURE_VISION_ENDPOINT = os.environ.get("AZURE_VISION_ENDPOINT", "")
            AZURE_OPENAI_API_KEY_OVERRIDE = os.environ.get("AZURE_OPENAI_API_KEY_OVERRIDE")
            
            # Used only with non-Azure OpenAI deployments
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION")
            OPENAI_CHATGPT_MODEL = os.environ.get("AZURE_OPENAI_CHATGPT_MODEL")
            
            AZURE_USE_AUTHENTICATION = os.environ.get("AZURE_USE_AUTHENTICATION", "").lower() == "true"
            AZURE_ENFORCE_ACCESS_CONTROL = os.environ.get("AZURE_ENFORCE_ACCESS_CONTROL", "").lower() == "true"
            AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS = os.environ.get("AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS", "").lower() == "true"
            AZURE_ENABLE_UNAUTHENTICATED_ACCESS = os.environ.get("AZURE_ENABLE_UNAUTHENTICATED_ACCESS", "").lower() == "true"
            AZURE_SERVER_APP_ID = os.environ.get("AZURE_SERVER_APP_ID")
            AZURE_SERVER_APP_SECRET = os.environ.get("AZURE_SERVER_APP_SECRET")
            AZURE_CLIENT_APP_ID = os.environ.get("AZURE_CLIENT_APP_ID")
            AZURE_AUTH_TENANT_ID = os.environ.get("AZURE_AUTH_TENANT_ID", AZURE_TENANT_ID)

            KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT", "content")
            KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE", "sourcepage")

            AZURE_SEARCH_QUERY_LANGUAGE = os.environ.get("AZURE_SEARCH_QUERY_LANGUAGE", "en-us")
            AZURE_SEARCH_QUERY_SPELLER = os.environ.get("AZURE_SEARCH_QUERY_SPELLER", "lexicon")
            AZURE_SEARCH_SEMANTIC_RANKER = os.environ.get("AZURE_SEARCH_SEMANTIC_RANKER", "free").lower()
            AZURE_SEARCH_QUERY_REWRITING = os.environ.get("AZURE_SEARCH_QUERY_REWRITING", "false").lower()
            AZURE_SEARCH_FIELD_NAME_EMBEDDING = os.environ.get("AZURE_SEARCH_FIELD_NAME_EMBEDDING", "embedding")

            AZURE_SPEECH_SERVICE_ID = os.environ.get("AZURE_SPEECH_SERVICE_ID")
            AZURE_SPEECH_SERVICE_LOCATION = os.environ.get("AZURE_SPEECH_SERVICE_LOCATION")
            AZURE_SPEECH_SERVICE_VOICE = os.environ.get("AZURE_SPEECH_SERVICE_VOICE", "en-US-AndrewMultilingualNeural")

            USE_MULTIMODAL = os.environ.get("USE_MULTIMODAL", "").lower() == "true"
            RAG_SEARCH_TEXT_EMBEDDINGS = os.environ.get("RAG_SEARCH_TEXT_EMBEDDINGS", "true").lower() == "true"
            RAG_SEARCH_IMAGE_EMBEDDINGS = os.environ.get("RAG_SEARCH_IMAGE_EMBEDDINGS", "true").lower() == "true"
            RAG_SEND_TEXT_SOURCES = os.environ.get("RAG_SEND_TEXT_SOURCES", "true").lower() == "true"
            RAG_SEND_IMAGE_SOURCES = os.environ.get("RAG_SEND_IMAGE_SOURCES", "true").lower() == "true"
            USE_USER_UPLOAD = os.environ.get("USE_USER_UPLOAD", "").lower() == "true"
            ENABLE_LANGUAGE_PICKER = os.environ.get("ENABLE_LANGUAGE_PICKER", "").lower() == "true"
            USE_SPEECH_INPUT_BROWSER = os.environ.get("USE_SPEECH_INPUT_BROWSER", "").lower() == "true"
            USE_SPEECH_OUTPUT_BROWSER = os.environ.get("USE_SPEECH_OUTPUT_BROWSER", "").lower() == "true"
            USE_SPEECH_OUTPUT_AZURE = os.environ.get("USE_SPEECH_OUTPUT_AZURE", "").lower() == "true"
            USE_CHAT_HISTORY_BROWSER = os.environ.get("USE_CHAT_HISTORY_BROWSER", "").lower() == "true"
            USE_CHAT_HISTORY_COSMOS = os.environ.get("USE_CHAT_HISTORY_COSMOS", "").lower() == "true"
            USE_AGENTIC_RETRIEVAL = os.environ.get("USE_AGENTIC_RETRIEVAL", "").lower() == "true"

            # Log the configuration for debugging
            current_app.logger.info("Azure OpenAI Configuration:")
            current_app.logger.info("  AZURE_OPENAI_SERVICE: %s", AZURE_OPENAI_SERVICE)
            current_app.logger.info("  AZURE_OPENAI_CHATGPT_DEPLOYMENT: %s", AZURE_OPENAI_CHATGPT_DEPLOYMENT)
            current_app.logger.info("  AZURE_OPENAI_CHATGPT_MODEL: %s", OPENAI_CHATGPT_MODEL)
            current_app.logger.info("  OPENAI_HOST: %s", OPENAI_HOST)
            
            # Create search clients
            search_client = None
            agent_client = None
            if AZURE_SEARCH_ENDPOINT:
                try:
                    if AZURE_SEARCH_INDEX:
                        search_client = SearchClient(
                            endpoint=AZURE_SEARCH_ENDPOINT, 
                            index_name=AZURE_SEARCH_INDEX, 
                            credential=azure_credential
                        )
                    if AZURE_SEARCH_AGENT:
                        agent_client = KnowledgeAgentRetrievalClient(
                            endpoint=AZURE_SEARCH_ENDPOINT, 
                            agent_name=AZURE_SEARCH_AGENT, 
                            credential=azure_credential
                        )
                except Exception:
                    current_app.logger.exception("Failed creating search/agent clients (continuing)")
            
            current_app.config[CONFIG_SEARCH_CLIENT] = search_client
            current_app.config[CONFIG_AGENT_CLIENT] = agent_client

            # Create blob manager
            global_blob_manager = None
            if AZURE_STORAGE_ACCOUNT:
                try:
                    global_blob_manager = BlobManager(
                        endpoint=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                        credential=azure_credential,
                        container=AZURE_STORAGE_CONTAINER,
                        image_container=AZURE_IMAGESTORAGE_CONTAINER,
                    )
                    current_app.config[CONFIG_GLOBAL_BLOB_MANAGER] = global_blob_manager
                except Exception:
                    current_app.logger.exception("Failed creating BlobManager (continuing)")

            # Setup authentication
            search_index = None
            if AZURE_USE_AUTHENTICATION and AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX:
                try:
                    search_index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=azure_credential)
                    search_index = await _safe_wait(
                        search_index_client.get_index(AZURE_SEARCH_INDEX), 
                        timeout=20, 
                        name="SearchIndexClient.get_index"
                    )
                    await _safe_wait(search_index_client.close(), timeout=5, name="SearchIndexClient.close")
                except Exception:
                    current_app.logger.exception("Error setting up SearchIndexClient (continuing)")

            auth_helper = AuthenticationHelper(
                search_index=search_index,
                use_authentication=AZURE_USE_AUTHENTICATION,
                server_app_id=AZURE_SERVER_APP_ID,
                server_app_secret=AZURE_SERVER_APP_SECRET,
                client_app_id=AZURE_CLIENT_APP_ID,
                tenant_id=AZURE_AUTH_TENANT_ID,
                require_access_control=AZURE_ENFORCE_ACCESS_CONTROL,
                enable_global_documents=AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS,
                enable_unauthenticated_access=AZURE_ENABLE_UNAUTHENTICATED_ACCESS,
            )
            current_app.config[CONFIG_AUTH_CLIENT] = auth_helper

            # Setup OpenAI client
            try:
                current_app.logger.info("OPENAI_HOST is %s, setting up %s OpenAI client", OPENAI_HOST, OPENAI_HOST)
                openai_client = await _safe_wait(
                    asyncio.to_thread(
                        setup_openai_client,
                        openai_host=OPENAI_HOST,
                        azure_credential=azure_credential,
                        azure_openai_api_version=AZURE_OPENAI_API_VERSION,
                        azure_openai_service=AZURE_OPENAI_SERVICE,
                        azure_openai_custom_url=AZURE_OPENAI_CUSTOM_URL,  # now only if explicitly set
                        azure_openai_api_key=AZURE_OPENAI_API_KEY_OVERRIDE,
                        openai_api_key=OPENAI_API_KEY,
                        openai_organization=OPENAI_ORGANIZATION,
                    ),
                    timeout=20,
                    name="setup_openai_client"
                )
                current_app.config[CONFIG_OPENAI_CLIENT] = openai_client
                if openai_client:
                    try:
                        # Extract the actual base URL for debugging
                        base_url = None
                        if hasattr(openai_client, 'base_url'):
                            base_url = openai_client.base_url
                        elif hasattr(openai_client, '_client') and hasattr(openai_client._client, 'base_url'):
                            base_url = openai_client._client.base_url
                            
                        current_app.logger.info(
                            "Azure OpenAI client initialized (base_url=%s, service=%s, deployment=%s, api_version=%s)",
                            base_url, AZURE_OPENAI_SERVICE, AZURE_OPENAI_CHATGPT_DEPLOYMENT, AZURE_OPENAI_API_VERSION
                        )
                    except Exception as e:
                        current_app.logger.warning("Could not extract OpenAI client details: %s", str(e))
                
                # Setup approaches after OpenAI client is ready
                if openai_client and search_client:
                    prompt_manager = PromptyManager()

                    ask_approach = RetrieveThenReadApproach(
                        search_client=search_client,
                        search_index_name=AZURE_SEARCH_INDEX,
                        openai_client=openai_client,
                        auth_helper=auth_helper,
                        chatgpt_model=OPENAI_CHATGPT_MODEL,
                        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                        agent_model=AZURE_OPENAI_SEARCHAGENT_MODEL,
                        agent_deployment=AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT,
                        agent_client=agent_client if agent_client else None,
                        embedding_model=os.environ.get("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002"),
                        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
                        embedding_dimensions=int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1536")),
                        embedding_field=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
                        sourcepage_field=KB_FIELDS_SOURCEPAGE,
                        content_field=KB_FIELDS_CONTENT,
                        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
                        query_speller=AZURE_SEARCH_QUERY_SPELLER,
                        prompt_manager=prompt_manager,
                    )
                    current_app.config[CONFIG_ASK_APPROACH] = ask_approach

                    chat_approach = ChatReadRetrieveReadApproach(
                        search_client=search_client,
                        search_index_name=AZURE_SEARCH_INDEX,
                        openai_client=openai_client,
                        auth_helper=auth_helper,
                        chatgpt_model=OPENAI_CHATGPT_MODEL,
                        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                        agent_model=AZURE_OPENAI_SEARCHAGENT_MODEL,
                        agent_deployment=AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT,
                        agent_client=agent_client if agent_client else None,
                        embedding_model=os.environ.get("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002"),
                        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
                        embedding_dimensions=int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1536")),
                        embedding_field=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
                        sourcepage_field=KB_FIELDS_SOURCEPAGE,
                        content_field=KB_FIELDS_CONTENT,
                        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
                        query_speller=AZURE_SEARCH_QUERY_SPELLER,
                        prompt_manager=prompt_manager,
                    )
                    current_app.config[CONFIG_CHAT_APPROACH] = chat_approach
                    # --- FIXED: semantic clustering enable block (indent + safety) ---
                    try:
                        metrics_store.configure_embeddings(
                            openai_client,
                            os.environ.get("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002")
                        )
                        current_app.logger.info("Semantic query clustering enabled for metrics.")
                    except Exception:
                        current_app.logger.warning("Failed to enable semantic query clustering (continuing).", exc_info=True)
                # -----------------------------------------------------------------
            except Exception:
                current_app.logger.exception("Failed setting up OpenAI client and approaches")

            # Setup user upload services if enabled
            if USE_USER_UPLOAD:
                try:
                    if AZURE_USERSTORAGE_ACCOUNT and AZURE_USERSTORAGE_CONTAINER:
                        user_blob_manager = AdlsBlobManager(
                            endpoint=f"https://{AZURE_USERSTORAGE_ACCOUNT}.dfs.core.windows.net",
                            container=AZURE_USERSTORAGE_CONTAINER,
                            credential=azure_credential,
                        )
                        current_app.config[CONFIG_USER_BLOB_MANAGER] = user_blob_manager
                        current_app.logger.info("User blob manager created successfully")
                        
                        # Setup file processors and ingester
                        file_processors = await _safe_wait(
                            asyncio.to_thread(
                                setup_file_processors,
                                azure_credential=azure_credential,
                                document_intelligence_service=os.environ.get("AZURE_DOCUMENTINTELLIGENCE_SERVICE"),
                                local_pdf_parser=os.environ.get("USE_LOCAL_PDF_PARSER", "").lower() == "true",
                                local_html_parser=os.environ.get("USE_LOCAL_HTML_PARSER", "").lower() == "true",
                                use_content_understanding=os.environ.get("USE_CONTENT_UNDERSTANDING", "").lower() == "true",
                                content_understanding_endpoint=os.environ.get("AZURE_CONTENTUNDERSTANDING_ENDPOINT"),
                                use_multimodal=USE_MULTIMODAL,
                                openai_client=openai_client,
                                openai_model=OPENAI_CHATGPT_MODEL,
                                openai_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT if OPENAI_HOST == OpenAIHost.AZURE else None,
                            ),
                            timeout=30,
                            name="setup_file_processors",
                        )
                        
                        if file_processors and search_client:
                            search_info = await _safe_wait(
                                setup_search_info(
                                    search_service=AZURE_SEARCH_SERVICE,
                                    index_name=AZURE_SEARCH_INDEX,
                                    azure_credential=azure_credential
                                ),
                                timeout=20,
                                name="setup_search_info"
                            )
                            
                            text_embeddings_service = await _safe_wait(
                                asyncio.to_thread(
                                    setup_embeddings_service,
                                    azure_credential=azure_credential,
                                    openai_host=OPENAI_HOST,
                                    emb_model_name=os.environ.get("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002"),
                                    emb_model_dimensions=int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1536")),
                                    azure_openai_service=AZURE_OPENAI_SERVICE,
                                    azure_openai_custom_url=AZURE_OPENAI_CUSTOM_URL,
                                    azure_openai_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
                                    azure_openai_api_version=AZURE_OPENAI_API_VERSION,
                                    azure_openai_key=clean_key_if_exists(AZURE_OPENAI_API_KEY_OVERRIDE),
                                    openai_key=clean_key_if_exists(OPENAI_API_KEY),
                                    openai_org=OPENAI_ORGANIZATION,
                                    disable_vectors=os.environ.get("USE_VECTORS", "").lower() == "false",
                                ),
                                timeout=20,
                                name="setup_embeddings_service"
                            )
                            
                            image_embeddings_service = None
                            if USE_MULTIMODAL:
                                image_embeddings_service = await _safe_wait(
                                    asyncio.to_thread(
                                        setup_image_embeddings_service,
                                        azure_credential=azure_credential,
                                        vision_endpoint=AZURE_VISION_ENDPOINT,
                                        use_multimodal=True,
                                    ),
                                    timeout=20,
                                    name="setup_image_embeddings_service"
                                )
                            
                            ingester = UploadUserFileStrategy(
                                search_info=search_info,
                                file_processors=file_processors,
                                embeddings=text_embeddings_service,
                                image_embeddings=image_embeddings_service,
                                search_field_name_embedding=AZURE_SEARCH_FIELD_NAME_EMBEDDING,
                                blob_manager=user_blob_manager,
                            )
                            current_app.config[CONFIG_INGESTER] = ingester
                            current_app.logger.info("Ingester created successfully")
                        
                except Exception:
                    current_app.logger.exception("Failed setting up user upload services")

            # Set feature flags
            current_app.config[CONFIG_MULTIMODAL_ENABLED] = USE_MULTIMODAL
            current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED] = AZURE_SEARCH_SEMANTIC_RANKER != "disabled"
            current_app.config[CONFIG_QUERY_REWRITING_ENABLED] = AZURE_SEARCH_QUERY_REWRITING == "true"
            current_app.config[CONFIG_REASONING_EFFORT_ENABLED] = os.environ.get("USE_REASONING_EFFORT", "").lower() == "true"
            current_app.config[CONFIG_STREAMING_ENABLED] = os.environ.get("USE_STREAMING", "true").lower() == "true"
            current_app.config[CONFIG_DEFAULT_REASONING_EFFORT] = os.environ.get("AZURE_OPENAI_REASONING_EFFORT", "medium")
            current_app.config[CONFIG_VECTOR_SEARCH_ENABLED] = os.environ.get("USE_VECTORS", "").lower() != "false"
            current_app.config[CONFIG_USER_UPLOAD_ENABLED] = USE_USER_UPLOAD
            current_app.config[CONFIG_LANGUAGE_PICKER_ENABLED] = ENABLE_LANGUAGE_PICKER
            current_app.config[CONFIG_SPEECH_INPUT_ENABLED] = USE_SPEECH_INPUT_BROWSER
            current_app.config[CONFIG_SPEECH_OUTPUT_BROWSER_ENABLED] = USE_SPEECH_OUTPUT_BROWSER
            current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED] = USE_SPEECH_OUTPUT_AZURE
            current_app.config[CONFIG_CHAT_HISTORY_BROWSER_ENABLED] = USE_CHAT_HISTORY_BROWSER
            current_app.config[CONFIG_CHAT_HISTORY_COSMOS_ENABLED] = USE_CHAT_HISTORY_COSMOS
            current_app.config[CONFIG_AGENTIC_RETRIEVAL_ENABLED] = USE_AGENTIC_RETRIEVAL
            current_app.config[CONFIG_RAG_SEARCH_TEXT_EMBEDDINGS] = RAG_SEARCH_TEXT_EMBEDDINGS
            current_app.config[CONFIG_RAG_SEARCH_IMAGE_EMBEDDINGS] = RAG_SEARCH_IMAGE_EMBEDDINGS
            current_app.config[CONFIG_RAG_SEND_TEXT_SOURCES] = RAG_SEND_TEXT_SOURCES
            current_app.config[CONFIG_RAG_SEND_IMAGE_SOURCES] = RAG_SEND_IMAGE_SOURCES

            # Setup speech services if enabled
            if USE_SPEECH_OUTPUT_AZURE:
                if not AZURE_SPEECH_SERVICE_ID or not AZURE_SPEECH_SERVICE_LOCATION:
                    current_app.logger.warning("Azure speech service not configured correctly, disabling")
                    current_app.config[CONFIG_SPEECH_OUTPUT_AZURE_ENABLED] = False
                else:
                    current_app.config[CONFIG_SPEECH_SERVICE_ID] = AZURE_SPEECH_SERVICE_ID
                    current_app.config[CONFIG_SPEECH_SERVICE_LOCATION] = AZURE_SPEECH_SERVICE_LOCATION
                    current_app.config[CONFIG_SPEECH_SERVICE_VOICE] = AZURE_SPEECH_SERVICE_VOICE
                    current_app.config[CONFIG_SPEECH_SERVICE_TOKEN] = None

            current_app.logger.info("Background initialization completed successfully")
            
        except Exception as exc:
            current_app.logger.exception("Unhandled exception in background initialization: %s", exc)
        finally:
            current_app.config["INIT_READY"] = True
            current_app.logger.info("INIT_READY set to True")

    # Create and start the background task
    task = asyncio.create_task(_background_init())
    current_app.config["_INIT_TASK"] = task
    current_app.logger.info("Spawned background initialization task; returning from before_app_serving quickly.")

@bp.after_app_serving
async def close_clients():
    """Clean up all async clients properly"""
    # Cancel background init task if still running
    if task := current_app.config.get("_INIT_TASK"):
        if not task.done():
            current_app.logger.info("Cancelling background initialization task")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    try:
        if search_client := current_app.config.get(CONFIG_SEARCH_CLIENT):
            await search_client.close()
    except Exception:
        current_app.logger.exception("Error closing search client")
        
    try:
        if agent_client := current_app.config.get(CONFIG_AGENT_CLIENT):
            await agent_client.close()
    except Exception:
        current_app.logger.exception("Error closing agent client")
        
    try:
        if blob_manager := current_app.config.get(CONFIG_GLOBAL_BLOB_MANAGER):
            await blob_manager.close_clients()
    except Exception:
        current_app.logger.exception("Error closing global blob manager")
        
    try:
        if user_blob_manager := current_app.config.get(CONFIG_USER_BLOB_MANAGER):
            await user_blob_manager.close_clients()
    except Exception:
        current_app.logger.exception("Error closing user blob manager")


# JSONEncoder and format_as_ndjson remain the same
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)

async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    """
    Consume an async generator of dict-like events and yield NDJSON lines.
    If an exception occurs while iterating, log it and yield a single error object (as JSON).
    """
    try:
        async for event in r:
            yield json.dumps(event, ensure_ascii=False, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps(error_dict(error), ensure_ascii=False) + "\n"

# ============= CUSTOM METRICS IMPLEMENTATION =============
from opentelemetry import trace, metrics
from datetime import datetime
import threading
from collections import defaultdict

# Initialize OpenTelemetry instrumentation
tracer = trace.get_tracer("azure-search-openai-demo", "1.0.0")
meter = metrics.get_meter("azure-search-openai-demo", "1.0.0")

# Thread-safe metrics storage
class MetricsStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_requests = 0
        self.unique_users = set()
        self.query_counts = defaultdict(int)
        self.index_access_counts = defaultdict(int)
        self.request_latencies = []
        self.last_reset = datetime.now()
        self.requests_by_endpoint = defaultdict(int)
        self.tokens_by_user = defaultdict(int)
        # --- NEW (semantic query clustering) ---
        self.embedding_fun = None          # coroutine: (text) -> list[float]
        self.semantic_clusters = []        # list[dict]: {"rep": str, "embedding": list[float], "count": int}
        self.semantic_threshold = 0.85     # cosine similarity threshold
        # ---------------------------------------

    def add_tokens(self, prompt_tokens=0, completion_tokens=0, total_tokens=0, user_id=None):
        with self.lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            if user_id:
                self.tokens_by_user[user_id] += total_tokens
    
    def add_request(self, endpoint, user_id=None):
        with self.lock:
            self.total_requests += 1
            self.requests_by_endpoint[endpoint] += 1
            if user_id:
                self.unique_users.add(user_id)
    
    # --- NEW helper to enable embeddings after OpenAI client is ready ---
    def configure_embeddings(self, openai_client, emb_model_name: str):
        """
        Configure the async embedding function. Safe to call multiple times.
        """
        async def _embed(text: str):
            if not text:
                return None
            try:
                resp = await openai_client.embeddings.create(
                    model=emb_model_name,
                    input=[text]
                )
                return resp.data[0].embedding
            except Exception:
                logging.debug("Embedding failed; falling back to literal counting", exc_info=True)
                return None
        self.embedding_fun = _embed

    # --- NEW internal async processor for semantic clustering ---
    async def _process_semantic(self, query: str):
        if not self.embedding_fun:
            return
        emb = await self.embedding_fun(query)
        if not emb:
            # Fallback: literal normalization
            normalized_query = query.lower().strip()[:100] if query else "empty"
            with self.lock:
                self.query_counts[normalized_query] += 1
            return
        # Compute cosine similarity against existing centroids
        def _cos(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        with self.lock:
            best = None
            best_sim = 0.0
            for c in self.semantic_clusters:
                sim = _cos(emb, c["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best = c
            if best and best_sim >= self.semantic_threshold:
                best["count"] += 1
                rep = best["rep"]
            else:
                rep = query
                self.semantic_clusters.append({"rep": rep, "embedding": emb, "count": 1})
            # Count representative so existing metrics (most_frequent_query) still work
            norm_rep = rep.lower().strip()[:100] if rep else "empty"
            self.query_counts[norm_rep] += 1

    def add_query(self, query):
        # Modified: if embeddings configured, cluster asynchronously; else original logic
        if not query:
            return
        if self.embedding_fun:
            # Fire and forget semantic processing (does not block request)
            asyncio.create_task(self._process_semantic(query))
            return
        # Original literal normalization path
        with self.lock:
            normalized_query = query.lower().strip()[:100] if query else "empty"
            self.query_counts[normalized_query] += 1

    def add_index_access(self, index_name):
        with self.lock:
            if index_name:
                self.index_access_counts[index_name] += 1
    
    def add_latency(self, latency_ms):
        with self.lock:
            self.request_latencies.append(latency_ms)
            # Keep only last 1000 latencies to prevent memory issues
            if len(self.request_latencies) > 1000:
                self.request_latencies = self.request_latencies[-1000:]
    
    def get_metrics(self):
        with self.lock:
            avg_latency = sum(self.request_latencies) / len(self.request_latencies) if self.request_latencies else 0
            most_frequent_query = max(self.query_counts.items(), key=lambda x: x[1])[0] if self.query_counts else "N/A"
            most_accessed_index = max(self.index_access_counts.items(), key=lambda x: x[1])[0] if self.index_access_counts else "N/A"
            # NEW: summarize semantic clusters (omit embeddings for size/privacy)
            semantic_clusters_summary = [
                {"rep": c["rep"], "count": c["count"]}
                for c in self.semantic_clusters
            ] if self.semantic_clusters else []
            return {
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_requests": self.total_requests,
                "unique_users_count": len(self.unique_users),
                "average_latency_ms": avg_latency,
                "most_frequent_query": most_frequent_query,
                "most_accessed_index": most_accessed_index,
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "top_token_users": sorted(self.tokens_by_user.items(), key=lambda x: x[1], reverse=True)[:10],
                # NEW
                "semantic_clusters": semantic_clusters_summary,
            }

# Initialize metrics store
metrics_store = MetricsStore()

# Create OpenTelemetry metrics instruments
request_counter = meter.create_counter(
    name="webapp.requests.total",
    description="Total number of requests to the web app",
    unit="1"
)

token_counter = meter.create_counter(
    name="openai.tokens.used",
    description="OpenAI tokens consumed",
    unit="tokens"
)

latency_histogram = meter.create_histogram(
    name="webapp.request.duration",
    description="Request duration in milliseconds",
    unit="ms"
)

# Observable gauges for aggregated metrics
from opentelemetry.metrics import Observation  # Add this import at the top with other metrics imports

# Then fix the callback functions:
def _observe_total_tokens(options):
    metrics_data = metrics_store.get_metrics()  # Renamed to avoid confusion
    yield Observation(
        metrics_data["total_tokens"], 
        {"app": "azure-search-openai-demo", "type": "total"}
    )
    yield Observation(
        metrics_data["prompt_tokens"], 
        {"app": "azure-search-openai-demo", "type": "prompt"}
    )
    yield Observation(
        metrics_data["completion_tokens"], 
        {"app": "azure-search-openai-demo", "type": "completion"}
    )

def _observe_unique_users(options):
    metrics_data = metrics_store.get_metrics()
    yield Observation(
        metrics_data["unique_users_count"], 
        {"app": "azure-search-openai-demo"}
    )

def _observe_average_latency(options):
    metrics_data = metrics_store.get_metrics()
    yield Observation(
        metrics_data["average_latency_ms"], 
        {"app": "azure-search-openai-demo"}
    )

# Register observable gauges
total_tokens_gauge = meter.create_observable_gauge(
    "openai.tokens.total",
    callbacks=[_observe_total_tokens],
    description="Total OpenAI tokens consumed",
    unit="tokens"
)

unique_users_gauge = meter.create_observable_gauge(
    "webapp.users.unique",
    callbacks=[_observe_unique_users],
    description="Number of unique users",
    unit="users"
)

average_latency_gauge = meter.create_observable_gauge(
    "webapp.latency.average",
    callbacks=[_observe_average_latency],
    description="Average request latency",
    unit="ms"
)

@bp.route("/metrics/custom", methods=["GET"])
@authenticated
async def custom_metrics(auth_claims: dict[str, Any]):
    """Endpoint to view current custom metrics (for debugging/monitoring)"""
    # Only allow admins or specific users to view metrics
    # You can customize this authorization logic
    metrics = metrics_store.get_metrics()
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "user": auth_claims.get("preferred_username", "unknown")
    })
# ============= END CUSTOM METRICS IMPLEMENTATION =============
def create_app():
    """Create and configure the Quart application"""
    app = Quart(__name__)
    
    # Set up logging first
    logging.basicConfig(level=logging.WARNING)
    app_level = os.environ.get("APP_LOG_LEVEL", "INFO")
    app.logger.setLevel(app_level)
    logging.getLogger("scripts").setLevel(app_level)
    
    # Initialize telemetry asynchronously (will be called during app startup)
    @app.before_serving
    async def init_telemetry():
        await _init_telemetry_once(app)
    
    # Register blueprints
    app.register_blueprint(bp)
    app.register_blueprint(chat_history_cosmosdb_bp)
    
    # Configure CORS if needed
    if allowed_origin := os.environ.get("ALLOWED_ORIGIN"):
        allowed_origins = allowed_origin.split(";")
        if len(allowed_origins) > 0:
            app.logger.info("CORS enabled for %s", allowed_origins)
            cors(app, allow_origin=allowed_origins, allow_methods=["GET", "POST"])
    
    app.logger.info("Quart app created successfully")
    return app