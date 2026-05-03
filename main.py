import streamlit as st
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# State definition
class State(TypedDict):
    user_data: dict
    fitness_plan: str
    feedback: str
    progress: List[str]
    messages: Annotated[list, add_messages]

# Utility function to get OpenAI LLM
def get_openai_llm():
    return ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini", temperature=0)
# Utility function to get Ollama LLM
def get_ollama_llm(model_name="tinyllama"):
    print(f"Creating ChatOllama with model: {model_name}")
    return ChatOllama(model=model_name)

# Instrucción común de idioma para uso en Costa Rica (América Latina)
LOCALE_ES_CR = (
    "Todas tus respuestas de texto deben estar en español (español de Costa Rica / América Latina). "
    "Usa tono cercano y respetuoso. Si mencionas alimentos o hábitos, puedes referirte a opciones comunes en la región cuando sea útil."
)

# User Input Agent
def user_input_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un asistente de entrenador físico con IA. Procesa la siguiente información del usuario:

        {user_input}

        {locale}

        Crea un perfil de usuario estructurado con todos los detalles relevantes para armar un plan de entrenamiento personalizado.
        Devuelve el perfil como una cadena JSON válida. Conserva los nombres de las claves en inglés (age, weight, height, gender, primary_goal, etc.) para compatibilidad con el sistema; los valores descriptivos deben estar en español."""
    )
    chain = prompt | llm | StrOutputParser()
    user_profile = chain.invoke({"user_input": json.dumps(state["user_data"]), "locale": LOCALE_ES_CR})
    try:
        state["user_data"] = json.loads(user_profile)
    except json.JSONDecodeError:
        pass
    state["messages"].append(AIMessage(content=f"Perfil de usuario procesado: {json.dumps(state['user_data'], indent=2, ensure_ascii=False)}"))
    return state
# routine generation agent
def routine_generation_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un entrenador físico con IA. Crea una rutina de ejercicio personalizada según estos datos del usuario:

        {user_data}

        {locale}

        Incluye un plan semanal detallado con:
        1. Tipos de ejercicios (nombres claros; si algo no es común en gimnasios caseros, indícalo)
        2. Duración y frecuencia de las sesiones
        3. Niveles de intensidad
        4. Días de descanso
        5. Recomendaciones alimentarias generales y prudentes (no sustituyen consejo médico)

        Presenta el plan de forma clara y estructurada, en español."""
    )
    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"user_data": json.dumps(state["user_data"], ensure_ascii=False), "locale": LOCALE_ES_CR})
    state["fitness_plan"] = plan
    state["messages"].append(AIMessage(content=f"Plan de entrenamiento generado: {plan}"))
    return state

# Feedback Collection Agent
def feedback_collection_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un asistente de entrenador físico con IA. Analiza la retroalimentación del usuario sobre su sesión o plan reciente:

        Plan de entrenamiento actual: {current_plan}
        Comentarios del usuario: {user_feedback}

        {locale}

        Resume lo que expresa el usuario y sugiere ajustes inmediatos posibles, en español."""
    )
    chain = prompt | llm | StrOutputParser()
    feedback_summary = chain.invoke(
        {"current_plan": state["fitness_plan"], "user_feedback": state["feedback"], "locale": LOCALE_ES_CR}
    )
    state["messages"].append(AIMessage(content=f"Análisis de retroalimentación: {feedback_summary}"))
    return state

# Routine Adjustment Agent
def routine_adjustment_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un entrenador físico con IA. Ajusta el plan de entrenamiento actual según la retroalimentación del usuario:

        Plan actual:
        {current_plan}

        Retroalimentación del usuario:
        {feedback}

        {locale}

        Entrega un plan semanal actualizado que atienda los comentarios del usuario y mantenga la estructura y los objetivos generales. Todo en español."""
    )
    chain = prompt | llm | StrOutputParser()
    updated_plan = chain.invoke(
        {"current_plan": state["fitness_plan"], "feedback": state["feedback"], "locale": LOCALE_ES_CR}
    )
    state["fitness_plan"] = updated_plan
    state["messages"].append(AIMessage(content=f"Plan de entrenamiento actualizado: {updated_plan}"))
    return state

# Progress Monitoring Agent
def progress_monitoring_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un seguimiento de progreso físico con IA. Revisa el avance del usuario y ofrece ánimo o sugerencias:

        Datos del usuario: {user_data}
        Plan actual: {current_plan}
        Historial de progreso: {progress_history}

        {locale}

        Resume el progreso, da ánimo y sugiere retos o ajustes nuevos si aplica. Todo en español."""
    )
    chain = prompt | llm | StrOutputParser()
    progress_update = chain.invoke(
        {
            "user_data": str(state["user_data"]),
            "current_plan": state["fitness_plan"],
            "progress_history": str(state["progress"]),
            "locale": LOCALE_ES_CR,
        }
    )
    state["progress"].append(progress_update)
    state["messages"].append(AIMessage(content=f"Actualización de progreso: {progress_update}"))
    return state

# Motivational Agent
def motivational_agent(state: State, llm):
    prompt = ChatPromptTemplate.from_template(
        """Eres un coach motivacional de fitness con IA. Ofrece ánimo, consejos prácticos o recordatorios al usuario:

        Datos del usuario: {user_data}
        Plan actual: {current_plan}
        Progreso reciente: {recent_progress}

        {locale}

        Genera un mensaje motivacional breve, un consejo útil o un recordatorio para mantener el compromiso con sus metas. En español."""
    )
    chain = prompt | llm | StrOutputParser()
    motivation = chain.invoke(
        {
            "user_data": str(state["user_data"]),
            "current_plan": state["fitness_plan"],
            "recent_progress": state["progress"][-1] if state["progress"] else "",
            "locale": LOCALE_ES_CR,
        }
    )
    state["messages"].append(AIMessage(content=f"Motivación: {motivation}"))
    return state

# AIFitnessCoach class
class AIFitnessCoach:
    def __init__(self):
        print("Initializing AIFitnessCoach")
        self.llm = get_openai_llm()
        # self.llm = get_ollama_llm() you can uncomment this if you prefer to use the locally running llms
        self.graph = self.create_graph()

    def create_graph(self):
        print("Creating graph")
        workflow = StateGraph(State)

        # Define nodes
        workflow.add_node("user_input", lambda state: user_input_agent(state, self.llm))
        workflow.add_node("routine_generation", lambda state: routine_generation_agent(state, self.llm))
        workflow.add_node("feedback_collection", lambda state: feedback_collection_agent(state, self.llm))
        workflow.add_node("routine_adjustment", lambda state: routine_adjustment_agent(state, self.llm))
        workflow.add_node("progress_monitoring", lambda state: progress_monitoring_agent(state, self.llm))
        workflow.add_node("motivation", lambda state: motivational_agent(state, self.llm))

        # Define edges
        workflow.add_edge("user_input", "routine_generation")
        workflow.add_edge("routine_generation", "feedback_collection")
        workflow.add_edge("feedback_collection", "routine_adjustment")
        workflow.add_edge("routine_adjustment", "progress_monitoring")
        workflow.add_edge("progress_monitoring", "motivation")
        workflow.add_edge("motivation", END)

        # Set entry point
        workflow.set_entry_point("user_input")
        print("Graph created")
        return workflow.compile()

    def run(self, user_input):
        print("Running AIFitnessCoach")
        initial_state = State(
            user_data=user_input,
            fitness_plan="",
            feedback="",
            progress=[],
            messages=[HumanMessage(content=json.dumps(user_input))]
        )
        print(f"Initial state: {initial_state}")
        final_state = self.graph.invoke(initial_state)
        print(f"Final state: {final_state}")
        return final_state["messages"]

def _etiqueta_mensaje(tipo: str) -> str:
    if tipo == "human":
        return "Usuario"
    if tipo == "ai":
        return "Asistente"
    return tipo.capitalize()


# Streamlit UI
def main():
    st.set_page_config(page_title="IA", layout="wide")
    st.title("AI Coach Fitness and Nutrition")
    st.caption("Crea y personaliza tu plan de entrenamiento con IA, utilizando la documentacion cientifica para crear los mejores planes de entrenamiento.")

    # Initialize session state
    if "fitness_coach" not in st.session_state:
        st.session_state.fitness_coach = AIFitnessCoach()

    tab1, tab2 = st.tabs(["Crear plan de entrenamiento", "Actualizar plan"])

    with tab1:
        st.header("Crea tu plan de entrenamiento personalizado")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Edad", min_value=1, max_value=120)
        with col2:
            weight = st.number_input("Peso (kg)", min_value=1.0)
        with col3:
            height = st.number_input("Estatura (cm)", min_value=1.0)
        
        gender = st.radio("Género", ["Masculino", "Femenino", "Otro"])
        
        primary_goal = st.selectbox(
            "Objetivo principal",
            ["Pérdida de peso", "Ganancia muscular", "Mejorar resistencia", "Condición física general"]
        )
        
        target_timeframe = st.selectbox(
            "Plazo objetivo",
            ["3 meses", "6 meses", "1 año"]
        )
        
        workout_preferences = st.multiselect(
            "Tipos de entrenamiento preferidos",
            ["Cardio", "Fuerza / musculación", "Yoga", "Pilates", "Flexibilidad", "HIIT"]
        )
        
        workout_duration = st.slider(
            "Duración preferida por sesión (minutos)",
            min_value=15,
            max_value=120,
            step=15
        )
        
        workout_days = st.multiselect(
            "Días preferidos para entrenar",
            ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        )
        
        activity_level = st.radio(
            "Nivel de actividad actual",
            ["Sedentario", "Ligeramente activo", "Moderadamente activo", "Muy activo"]
        )
        
        health_conditions = st.text_area("Condiciones de salud o lesiones")
        dietary_preferences = st.text_area("Preferencias alimentarias (opcional)")
        
        if st.button("Generar plan de entrenamiento"):
            user_data = {
                "age": age,
                "weight": weight,
                "height": height,
                "gender": gender,
                "primary_goal": primary_goal,
                "target_timeframe": target_timeframe,
                "workout_preferences": workout_preferences,
                "workout_duration": workout_duration,
                "workout_days": workout_days,
                "activity_level": activity_level,
                "health_conditions": health_conditions,
                "dietary_preferences": dietary_preferences
            }
            
            with st.spinner("Generando tu plan personalizado..."):
                messages = st.session_state.fitness_coach.run(user_data)
                st.session_state.last_plan = messages
                
            for message in messages:
                st.write(f"**{_etiqueta_mensaje(message.type)}:** {message.content}")

    with tab2:
        st.header("Actualizar tu plan de entrenamiento")
        feedback = st.text_area("Comentarios sobre tu plan actual:")
        
        if st.button("Actualizar plan"):
            with st.spinner("Actualizando tu plan..."):
                messages = st.session_state.fitness_coach.run({"feedback": feedback})
                
            for message in messages:
                st.write(f"**{_etiqueta_mensaje(message.type)}:** {message.content}")

if __name__ == "__main__":
    main()
