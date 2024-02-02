import src.app as app


def test_app():
    # Test case 1: User is authenticated
    app.authentication_status = True
    app.user_prompt = "Hello, how are you?"
    app.llm_chain = MockLLMChain()  # Replace with your own mock implementation
    app.st.session_state.messages = []

    app.run_chat()

    assert len(app.st.session_state.messages) == 1
    assert app.st.session_state.messages[0]["role"] == "assistant"
    assert app.st.session_state.messages[0]["content"] == "Mock response"

    # Test case 2: User is not authenticated
    app.authentication_status = False
    app.user_prompt = "Hello, how are you?"
    app.st.session_state.messages = []

    app.run_chat()

    assert len(app.st.session_state.messages) == 0
    assert app.st._last_alert["message"] == "please enter the password."


if __name__ == "__main__":
    test_app()
