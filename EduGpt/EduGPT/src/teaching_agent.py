import os
from typing import Any, Dict, List
from dotenv import load_dotenv

# ✅ Correct imports for LangChain 0.1.x
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel as BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ✅ Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# ✅ Conversation chain for teaching
class InstructorConversationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Create the chain with the teaching prompt."""
        instructor_agent_inception_prompt = """
        As a Machine Learning instructor agent, your task is to teach the user based on a provided syllabus.
        The syllabus serves as a roadmap for the learning journey, outlining the specific topics, concepts, and learning objectives to be covered.
        Review the provided syllabus carefully. Follow the exact order of topics — do not skip or reorder them.
        Explain each concept clearly and progressively, step by step, ensuring the user understands.
        
        Following '===' is the syllabus about {topic}.
        ===
        {syllabus}
        ===
        
        Use this syllabus to teach the user about {topic}.
        Maintain a supportive tone and adapt to the user's learning pace.
        Generate only one teaching step per response. 
        When done with your current teaching step, end with '<END_OF_TURN>'.
        ===
        {conversation_history}
        ===
        """

        prompt = PromptTemplate(
            template=instructor_agent_inception_prompt,
            input_variables=["syllabus", "topic", "conversation_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# ✅ Teaching Controller Agent
class TeachingGPT(Chain):
    """Controller model for the Teaching Agent."""

    syllabus: str = ""
    conversation_topic: str = ""
    conversation_history: List[str] = []
    teaching_conversation_utterance_chain: InstructorConversationChain = Field(...)

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self, syllabus, task):
        """Initialize the teaching agent with the syllabus and topic."""
        self.syllabus = syllabus
        self.conversation_topic = task
        self.conversation_history = []

    def human_step(self, human_input: str):
        """Add a human message to the conversation."""
        human_input = human_input.strip() + " <END_OF_TURN>"
        self.conversation_history.append(f"Human: {human_input}")

    def instructor_step(self):
        """Generate the instructor's next teaching message."""
        return self._call_instructor({})

    def _call(self, inputs: Dict[str, Any]) -> None:
        pass

    def _call_instructor(self, inputs: Dict[str, Any]) -> str:
        """Invoke the Gemini LLM chain to produce the instructor's message."""
        ai_message = self.teaching_conversation_utterance_chain.run(
            syllabus=self.syllabus,
            topic=self.conversation_topic,
            conversation_history="\n".join(self.conversation_history),
        )
        self.conversation_history.append(f"Instructor: {ai_message}")
        print("Instructor:", ai_message.rstrip("<END_OF_TURN>"))
        return ai_message

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "TeachingGPT":
        """Factory method to build the full teaching agent."""
        chain = InstructorConversationChain.from_llm(llm, verbose=verbose)
        return cls(teaching_conversation_utterance_chain=chain, verbose=verbose, **kwargs)


# ✅ Initialize the Gemini-powered teaching agent
config = dict(conversation_history=[], syllabus="", conversation_topic="")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
teaching_agent = TeachingGPT.from_llm(llm, verbose=False, **config)
