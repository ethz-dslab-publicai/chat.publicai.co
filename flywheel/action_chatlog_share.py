"""
Chatlog Data Contribution - Action
==================================

Enables a user to contribute their chatlog to PublicAI's dataset.

Responsible for preprocessing chat-to-be-shared for PII, requesting the user's
privacy choices and confirmation from the frontend, and submitting the final
contribution to the dataset.
"""

from datetime import datetime, timezone
from uuid import uuid4

# from pydantic import BaseModel
# from open_webui.models.feedbacks import Feedbacks

# TODO: Set up proper logging

class Action:

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __id__=None,
    ) -> dict:

        try:

            # TODO: Process chat log for PII stripping

            print("\n\n\nABOUT TO SEND DATA CONTRIBUTION EVENT\n\n\n")
            form_response = await __event_call__(
                {
                    "type": "data_contribution",
                    "data": {
                        "contribution_id": str(uuid4()),
                    }
                }
            )
            print("RESPONSE", form_response, type(form_response))

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

            # Otherwise, parse user privacy choices from form_response and
            # submit the contribution accordingly.
            contribution_status["status"] = "submitted"
            contribution_status["contribution_id"] = 'dummy' # TODO: Use hash of conversation data

            # TODO: Parse user privacy choices from form_response

            # if include_feedback is True:
            #     Feedbacks.get_feedbacks_by_user_id

            print("SUBMITTING DATA CONTRIBUTION")
            contribution_status["timestamp"] = datetime.now(timezone.utc).isoformat()
            body["messages"][-1]["contribution_status"] = contribution_status


        except Exception as e:
            print("ERROR PROCESSING DATA CONTRIBUTION", e)

        return body
