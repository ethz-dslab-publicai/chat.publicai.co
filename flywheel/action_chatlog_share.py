"""
title: Contribute Chatlog
author: PublicAI.co and Parker Addison
version: 1.0.0
required_open_webui_version: 0.6.0
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM2NzY3NjciIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNNCAxMnY3YzAgLjU1LjQ1IDEgMSAxaDE0Yy41NSAwIDEtLjQ1IDEtMXYtNyIvPjxwYXRoIGQ9Ik0xMiAxNlYzIi8+PHBhdGggZD0iTTggN2w0LTQgNCA0Ii8+PC9zdmc+
requirements: presidio_analyzer,presidio_anonymizer,huggingface_hub

Chatlog Data Contribution - Action
==================================

Enables a user to contribute their chatlog to PublicAI's dataset.

Responsible for preprocessing chat-to-be-shared for PII, requesting the user's
privacy choices and confirmation from the frontend, and submitting the final
contribution to the dataset.
"""

import io
import json
from copy import deepcopy
from datetime import datetime, timezone
import re
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi
from open_webui.models.feedbacks import Feedbacks
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field

PII_ENTITIES = [
    'IBAN_CODE',
    'US_PASSPORT',
    'UK_NHS',
    'PHONE_NUMBER',
    'IP_ADDRESS',
    # 'NRP',
    # 'LOCATION',
    'PERSON',
    'EMAIL_ADDRESS',
    'CRYPTO',
    'URL',
    'MEDICAL_LICENSE',
    'US_DRIVER_LICENSE',
    'US_ITIN',
    'US_SSN',
    'CREDIT_CARD',
    'US_BANK_NUMBER',
    # 'DATE_TIME'
]

CHAT_FIELDS_TO_KEEP = [
    "model",
    "messages",
]

MESSAGE_FIELDS_TO_KEEP = [
    "role",
    "content",
    "timestamp",
    "contribution_status",
    # TODO: Support tool calls and redacted attachments
]

FEEDBACK_FIELDS_TO_KEEP = [
    "rating",
    "model_id",
    "reason",
    "comment",
    "tags",
]

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

class FormResponse(BaseModel):
    identification_type: str
    user_id: str | None
    huggingface_token: str | None
    include_thumbs: bool
    rating_accuracy: int | None
    rating_instructions: int | None
    rating_tags: list[str]
    usage_model_training: bool
    usage_qa: bool
    usage_analytics: bool
    usage_other: bool
    usage_other_text: str
    comments: str

class Contribution(BaseModel):
    contribution_id: str
    clean_messages: list[dict[str, Any]]
    model: str
    license_opts: dict[str, bool]
    license_other: str | None
    attribution_mode: Literal["anonymous", "huggingface"]
    attribution: Literal["anonymous"] | str
    ratings: dict | None
    comment: str | None
    contributed_at: str


class Action:

    class Valves(BaseModel):
        default_hf_token: str = Field(
            default="", description="Service HF token used by the app to open PRs"
        )
        dataset_repo: str = Field(
            default="", description="Dataset repo (owner/name)"
        )
        min_messages: int = Field(
            default=2, description="Minimum messages required to share"
        )
        max_messages: int = Field(
            default=100, description="Maximum messages allowed per share"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.shared_chat_ids = set()

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __id__=None,
    ) -> dict:

        try:

            # Show progress for processing
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Preparing your chatlog for sharing (removing sensitive info)...",
                        "done": False,
                    },
                }
            )

            # Last message ID is used as the contribution ID. Has edge case if
            # the user manually edits previous LLM messages.
            contribution_id = body["messages"][-1]["id"]

            # Limit number of messages to share
            body_to_share = deepcopy(body)
            num_messages = len(body.get("messages", []))
            num_to_trim = max(0, num_messages - self.valves.max_messages)
            if num_to_trim > 0:
                body_to_share["messages"] = body_to_share["messages"][num_to_trim:]

            print("\n\n\nRECEIVED DATA CONTRIBUTION REQUEST\n\n\n")
            print(body_to_share)

            # Perform PII analysis and redaction on the chatlog
            print("\n\n\nCLEANING CHATLOG\n\n\n")
            pii_counts_per_message, redacted_chatlog = clean_chatlog(body_to_share)
            # print("PII COUNTS PER MESSAGE", pii_counts_per_message)

            # Add feedback to redacted chatlog
            feedback = Feedbacks.get_feedbacks_by_user_id(
                user_id=__user__.get("id")
            )
            # Filter to only this chat ID
            feedback = [
                fb for fb in feedback if fb.meta["chat_id"] == body["chat_id"]
            ]
            # Use message index to insert feedback into the correct message.
            for fb in feedback:
                # Adjust for trimmed messages from the start. Index is 1-based.
                message_index = fb.meta["message_index"] - 1 - num_to_trim

                # Add feedback to the message, only keeping necessary fields
                redacted_chatlog["messages"][message_index]["feedback"] = {
                    k: v for k, v in fb.data.items() if k in FEEDBACK_FIELDS_TO_KEEP
                }

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Preparing your chatlog for sharing (removing sensitive info)",
                        "done": True,
                    },
                }
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Asking for your privacy choices",
                        "done": False,
                    },
                }
            )

            print("\n\n\nSENDING CONTRIBUTION FORM TO USER\n\n\n")
            form_response = None
            try:
                # Replace `<ENTITY_NAME>` with `==<ENTITY_NAME>==` for
                # highlighting
                redacted_chatlog_to_share = json.dumps(redacted_chatlog)
                for entity in PII_ENTITIES:
                    pattern = re.compile(r'<'+entity+r'>')
                    redacted_chatlog_to_share = pattern.sub(r'==<' + entity + r'>==', redacted_chatlog_to_share)
                form_response = await __event_call__(
                    {
                        "type": "data_contribution",
                        "data": {
                            "contribution_id": str(uuid4()),
                            "redacted_chatlog": redacted_chatlog_to_share,
                            "pii_counts_per_message": json.dumps(pii_counts_per_message),
                        }
                    }
                )
            except Exception as e:
                print("ERROR GETTING FORM RESPONSE", e)
                # await __event_emitter__({
                #     "type": "notification",
                #     "data": {
                #         "type": "error",
                #         "content": f"Error waiting for form response: {e}",
                #     }
                # })

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Asking for your privacy choices",
                        "done": True,
                    },
                }
            )

            print("\n\n\nPROCESSING CONTRIBUTION FORM RESPONSE\n\n\n")
            # Add a contribution status to the last message so we can inform the
            # user of the result, and keep track of contributions in chat logs.
            contribution_status = {}

            # IMPORTANT: If form_response is False then the user cancelled. Stop
            # all further processing and just return the with cancelled status.
            if not form_response:
                print("USER CANCELLED DATA CONTRIBUTION")
                contribution_status["status"] = "cancelled"
                contribution_status["timestamp"] = datetime.now(timezone.utc).isoformat()
                body["messages"][-1]["contribution_status"] = contribution_status
                return body

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Submitting your data contribution",
                        "done": False,
                    },
                }
            )
            # Otherwise, parse the user privacy choices from the form, and
            # submit the contribution accordingly.
            form_response = FormResponse(**form_response)
            # print("PARSED RESPONSE", form_response)

            # Parse user privacy choices from form_response
            if form_response.identification_type == "anonymous":
                form_response.user_id = "anonymous"

            license_opts = {
                "model_training": form_response.usage_model_training,
                "qa": form_response.usage_qa,
                "analytics": form_response.usage_analytics,
            }
            if form_response.usage_other:
                license_opts_other = form_response.usage_other_text
            else:
                license_opts_other = None

            ratings = {}
            if form_response.rating_accuracy is not None:
                ratings["accuracy"] = form_response.rating_accuracy
            if form_response.rating_instructions is not None:
                ratings["instructions"] = form_response.rating_instructions
            if form_response.rating_tags:
                ratings["tags"] = form_response.rating_tags

            if form_response.include_thumbs:
                # Use message index to insert feedback into the correct message.
                for fb in feedback:
                    # Adjust for trimmed messages from the start. Index is 1-based.
                    message_index = fb.meta["message_index"] - 1 - num_to_trim
                    # Add feedback to the message, only keeping necessary fields
                    redacted_chatlog["messages"][message_index]["feedback"] = {
                        k: v for k, v in fb.data.items() if k in FEEDBACK_FIELDS_TO_KEEP
                    }

            contributed_at = datetime.now(timezone.utc).isoformat()


            contribution = Contribution(
                contribution_id=contribution_id,
                clean_messages=redacted_chatlog["messages"],
                model=redacted_chatlog.get("model", "unknown"),
                license_opts=license_opts,
                license_other=license_opts_other,
                attribution_mode=form_response.identification_type,
                attribution=form_response.user_id,
                ratings=ratings or None,
                comment=form_response.comments or None,
                contributed_at=contributed_at
            )

            print("SUBMITTING DATA CONTRIBUTION", contribution)

            pr_url = create_hf_pr(
                contribution=contribution,
                hf_token=form_response.huggingface_token
                or self.valves.default_hf_token,
                dataset_repo=self.valves.dataset_repo,
            )
            print("CREATED PR", pr_url)

            contribution_status["status"] = "submitted"
            contribution_status["contribution_id"] = contribution_id
            contribution_status["pr_url"] = pr_url
            contribution_status["timestamp"] = contributed_at
            body["messages"][-1]["contribution_status"] = contribution_status

            # self.shared_chat_ids.add(contribution_id)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Submitting your data contribution",
                        "done": True,
                    },
                }
            )
            await __event_emitter__(
                {
                    "type": "notification",
                    "data": {
                        "type": "success",
                        "content": "Your data contribution has been submitted successfully!",
                    },
                }
            )

            return body

        except Exception as e:
            # await __event_emitter__({
            #     "type": "notification",
            #     "data": {
            #         "type": "error",
            #         "content": f"Error processing data contribution: {e}",
            #     }
            # })
            print("ERROR PROCESSING DATA CONTRIBUTION", type(e), e)

        return body


# ---------------------------------- Helpers --------------------------------- #

def clean_chatlog(
    chatlog: dict
) -> tuple[dict, dict]:
    """
    Takes a chatlog dictionary and returns a dictionary describing and PII found
    in the messages, as well as a copy of the chatlog with PII redacted and only
    the necessary fields kept.
    """

    pii_counts_per_message = {}
    redacted_chatlog = deepcopy(chatlog)

    # Keep only necessary chatlog fields
    redacted_chatlog = {k: v for k, v in redacted_chatlog.items() if k in CHAT_FIELDS_TO_KEEP}

    for i, message in enumerate(chatlog.get("messages", [])):

        # Keep only necessary message fields
        redacted_chatlog["messages"][i] = {k: v for k, v in message.items() if k in MESSAGE_FIELDS_TO_KEEP}

        # Redact PII from message content
        raw_content = message.get("content", "").strip()
        if raw_content:
            pii_counts, content = _redact_message_pii(raw_content)
            pii_counts_per_message[i] = pii_counts
            redacted_chatlog["messages"][i]["content"] = content
        else:
            continue

    return pii_counts_per_message, redacted_chatlog


def _redact_message_pii(
    message: dict
) -> tuple[dict, dict]:
    """
    Takes a message dictionary and returns a dictionary showing the number of
    each PII entity found in the message, as well as a copy of the message with
    PII redacted.
    """

    pii_analysis = analyzer.analyze(
        text=message,
        entities=PII_ENTITIES,
        language='en',  # TODO: Can be made dynamic based on detecting chat language
    )
    anon_message = anonymizer.anonymize(
        text=message,
        analyzer_results=pii_analysis,
    )
    entity_types, counts = np.unique(
        [item.entity_type for item in pii_analysis],
        return_counts=True
    )
    pii_counts = {str(entity): int(count) for entity, count in zip(entity_types, counts)}

    anon_text = anon_message.text

    return pii_counts, anon_text

def create_hf_pr(
    contribution: Contribution,
    hf_token: str,
    dataset_repo: str,
) -> str:
    """
    Creates a HuggingFace PR with the contributed chatlog data. Returns the URL
    of the pull request.
    """

    assert hf_token, "HuggingFace token is required to create PR"
    assert dataset_repo, "Dataset repo is required to create PR"

    api = HfApi()
    file_path = f"contributions/{contribution.contribution_id}.json"
    json_contribution = contribution.model_dump_json()

    pr_title = f"Data Contribution - {contribution.contribution_id}"
    pr_description = f"Contributed by: {contribution.attribution}"

    commit_info = api.create_commit(
        repo_id=dataset_repo,
        token=hf_token,
        operations=[
            CommitOperationAdd(
                path_in_repo=file_path,
                path_or_fileobj=io.BytesIO(json_contribution.encode()),
            )
        ],
        commit_message=pr_title,
        commit_description=pr_description,
        create_pr=True,
        repo_type="dataset",
    )
    pr_num = getattr(commit_info, "pr_num", None)
    if pr_num is not None:
        pr_url = f"https://huggingface.co/datasets/{dataset_repo}/discussions/{pr_num}"
    else:
        pr_url = f"https://huggingface.co/datasets/{dataset_repo}"

    return pr_url
