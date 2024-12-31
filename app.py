from dotenv import dotenv_values
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from langgraph.prebuilt import create_react_agent # type: ignore
from typing import Optional, List
from tools.AirQuality import get_air_quality
from tools.RagFAQ import get_rag_faq

# Load api key
config = dotenv_values(".env") 
OPENAI_API_KEY = config['OPENAI_API_KEY']


def create_agent():
    
    # Memory with session ID
    memory = MemorySaver()
    # Load model (gpt-4o)
    model = ChatOpenAI(model="gpt-4o",verbose = True)
    # Load tools
    tools = [get_air_quality, get_rag_faq]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    
    return agent_executor

def run_agent(history: Optional[List[BaseMessage]] = None):
    agent = create_agent()
    print('Agent is now running')

    # Configuration for session ID
    config = {"configurable": {"thread_id": 'tester'}}

    # 如果有提供歷史消息，將它們更新到代理的內存中
    if history:
        agent.update_state(config, {"messages": history})
    
    sys_prompt = '你是環保局QA小幫手負責協助民眾問題，回答皆使用繁體中文'
    SystemMessage(content=sys_prompt)
    agent.update_state(config, {"messages": sys_prompt})

    while True:
        question = input("Please enter your question (or 'Q' to quit): ")
        if question.upper() == "Q":
            break

        # Invoke the agent with the provided session ID
        response = agent.invoke({"messages": [HumanMessage(content=question)]}, config)
        if response["messages"][-1].tool_calls:
            print(response["messages"][-1].tool_calls)
        print(response["messages"][-1].content)  # Print the agent's response
        
if __name__=="__main__":
    run_agent()