from django.shortcuts import render, redirect

from .models import Chat, Document, Message

from rag_service import answer_question
from ingestion_service import ingest_pdf


def home(request):

    # ==========================================
    # GET ALL CHATS
    # ==========================================

    chats = Chat.objects.all().order_by("-updated_at")

    chat_id = request.GET.get("chat")

    current_chat = None

    # ==========================================
    # LOAD CURRENT CHAT
    # ==========================================

    if chat_id:

        try:
            current_chat = Chat.objects.get(
                id=chat_id
            )

        except Chat.DoesNotExist:
            current_chat = None

    error = None

    # ==========================================
    # HANDLE POST
    # ==========================================

    if request.method == "POST":

        question = request.POST.get(
            "question",
            ""
        ).strip()

        uploaded_file = request.FILES.get("file")

        chat_id = request.POST.get("chat_id")

        # ==========================================
        # GET OR CREATE CHAT
        # ==========================================

        if chat_id:

            try:

                current_chat = Chat.objects.get(
                    id=chat_id
                )

            except Chat.DoesNotExist:

                current_chat = Chat.objects.create(
                    title="New Chat"
                )

        else:

            current_chat = Chat.objects.create(
                title="New Chat"
            )

        # ==========================================
        # PROCESS PDF IF ONE WAS ATTACHED
        # ==========================================

        document = None

        if uploaded_file:

            try:

                # ----------------------------------
                # DOCUMENT TITLE
                # ----------------------------------

                document_title = uploaded_file.name.rsplit(
                    ".",
                    1
                )[0]

                # ----------------------------------
                # SAVE DOCUMENT
                # ----------------------------------

                document = Document.objects.create(
                    chat=current_chat,
                    title=document_title,
                    file=uploaded_file
                )

                print(
                    "PDF uploaded:",
                    document.file.path
                )

                # ----------------------------------
                # INGEST PDF
                # ----------------------------------

                total_chunks = ingest_pdf(
                    file_path=document.file.path,
                    document_name=document.title,
                    chat_id=current_chat.id
                )

                print(
                    "PDF ingestion completed."
                )

                print(
                    "Total chunks:",
                    total_chunks
                )

                print(
                    "Chat ID:",
                    current_chat.id
                )

                # ----------------------------------
                # IF THIS IS FIRST DOCUMENT,
                # USE ITS NAME AS CHAT TITLE
                # ----------------------------------

                if current_chat.title == "New Chat":

                    current_chat.title = document_title

                    current_chat.save()

            except Exception as e:

                print(
                    "UPLOAD/INGESTION ERROR:",
                    str(e)
                )

                error = (
                    "Error while processing the PDF: "
                    + str(e)
                )

        # ==========================================
        # PROCESS QUESTION
        # ==========================================

        if question:

            try:

                print(
                    "Question received:",
                    question
                )

                # ----------------------------------
                # SAVE USER MESSAGE
                # ----------------------------------

                if document:

                    message_content = (
                        f"📎 {document.title}.pdf\n\n"
                        f"{question}"
                    )

                else:

                    message_content = question

                Message.objects.create(
                    chat=current_chat,
                    role="user",
                    content=message_content
                )

                # ----------------------------------
                # GENERATE AI ANSWER
                # ----------------------------------

                answer = answer_question(
                    question,
                    chat_id=current_chat.id
                )

                print(
                    "AI answer received:",
                    answer
                )

                # ----------------------------------
                # SAVE AI MESSAGE
                # ----------------------------------

                Message.objects.create(
                    chat=current_chat,
                    role="assistant",
                    content=answer
                )

                # ----------------------------------
                # UPDATE CHAT
                # ----------------------------------

                current_chat.save()

                # ----------------------------------
                # REDIRECT
                # ----------------------------------

                return redirect(
                    f"/?chat={current_chat.id}"
                )

            except Exception as e:

                print(
                    "RAG ERROR:",
                    str(e)
                )

                error = (
                    "Error while generating answer: "
                    + str(e)
                )

        elif uploaded_file and not error:

            # PDF was uploaded without a question.
            # Keep the user inside the chat.

            current_chat.save()

            return redirect(
                f"/?chat={current_chat.id}"
            )

    # ==========================================
    # CURRENT CHAT DOCUMENTS
    # ==========================================

    documents = []

    if current_chat:

        documents = (
            current_chat.documents
            .all()
            .order_by("-uploaded_at")
        )

    # ==========================================
    # CURRENT CHAT MESSAGES
    # ==========================================

    messages = []

    if current_chat:

        messages = (
            current_chat.messages
            .all()
            .order_by("created_at")
        )

    # ==========================================
    # REFRESH CHAT LIST
    # ==========================================

    chats = Chat.objects.all().order_by("-updated_at")

    # ==========================================
    # RENDER
    # ==========================================

    return render(
        request,
        "index.html",
        {
            "chats": chats,
            "current_chat": current_chat,
            "documents": documents,
            "messages": messages,
            "error": error,
        }
    )