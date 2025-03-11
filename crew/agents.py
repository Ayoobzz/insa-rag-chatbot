from crewai import Agent

main_chatbot = Agent(
    name="University Chatbot",
    role="Student Assistant",
    goal="Answer students' questions about the university",
    backstory="A helpful assistant designed to guide students."
)
