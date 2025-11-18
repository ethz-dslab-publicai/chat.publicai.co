Data Flywheel for Chat Logs and Feedback
========================================

This directory holds Open WebUI Functions that enable processing of voluntary data contributions for users to share their chat logs to PublicAI's dataset, for model training and improvement.

## What it does

When a user on chat.publicai.co receives a response, they will see a new button to share their chatlogs with feedback. When clicking the button, the user will see a form allowing them to set their privacy preferences for the contribution, as well as review the chats before they are sent.

## Architecture

The chatlog data share functionality uses two main components:

1. **A Python OWUI Action** _(in this repository)_

    The action defines the button and is responsible for processing the chatlogs once a user has selected their privacy options and approved the contribution. This processing takes place on the PublicAI-hosted OWUI server where the chats are processed, and no data leaves the server until the action submits a dataset pull request.

2. **Customizations to the OWUI Frontend** _(in the fork of open-webui)_

    The UI form shown to the user to set their privacy options is written as customized frontend code within the OWUI repository. The modal is shown when the chatlog action sends an _event_ to the frontend, and the user's form choices are returned to the action to finalize processing.
