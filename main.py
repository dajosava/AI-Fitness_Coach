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
        Devuelve el perfil como una cadena JSON válida. Conserva exactamente estas claves en inglés (incluye null donde falte dato):
        age, weight, height, gender, primary_goal, target_timeframe, workout_preferences, workout_duration, workout_days, activity_level,
        health_conditions, dietary_preferences,
        body_fat_percent, waist_cm, hip_cm, time_without_training, previous_training_types, previous_training_notes, why_stopped_training,
        equipment_available, equipment_notes, training_time_of_day, hours_per_week_total,
        max_pullups, max_pushups, can_run_1km_nonstop, resting_heart_rate,
        job_activity_type, stress_level_1_5, sleep_hours_avg,
        injury_status, movement_limitations,
        trains_solo_or_group, routine_style_preference, exercises_disliked, past_success_methods,
        meals_per_day, calorie_intention, supplements,
        target_weight_change_kg, goal_event_note, goal_focus_priorities,
        barrier_why_not_before, barrier_plan_adherence, motivation_scale_1_10.
        Los valores descriptivos deben estar en español."""
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

        Usa TODOS los datos disponibles (si hay equipamiento, horas/semana, horario del día, % grasa o perímetros, historial de desentrenamiento,
        lesiones o limitaciones de movimiento, métricas base, estrés/sueño/trabajo sedentario, motivación 1-10 y barreras, odio a ciertos ejercicios, etc.).
        El plan debe ser realista con el equipamiento declarado (si solo hay peso corporal, no bases el plan en máquinas de gimnasio completo).
        Ajusta volumen e intensidad si hay alto estrés, mal sueño o lesión activa. Respeta limitaciones explícitas (evita movimientos prohibidos).
        Si la motivación es baja o hay barreras claras, sugiere estrategias de adherencia concretas.

        Incluye un plan semanal detallado con:
        1. Tipos de ejercicios acordes al equipamiento y limitaciones (nombres claros)
        2. Duración y frecuencia alineadas a las horas/semana y días disponibles; considerá el horario del día si afecta energía
        3. Niveles de intensidad prudentes según nivel, desentrenamiento y salud
        4. Días de descanso
        5. Recomendaciones alimentarias acordes a comidas/déficit-superávit si constan; prudentes (no sustituyen consejo médico)

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
    st.markdown(
        """
        <style>
        /* Botones primarios: fondo primaryColor (amarillo) → texto e iconos negros */
        div[data-testid="stButton"] button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            color: #000000 !important;
        }
        div[data-testid="stButton"] button[kind="primary"] p,
        div[data-testid="stButton"] button[kind="primary"] span {
            color: #000000 !important;
        }
        div[data-testid="stButton"] button[kind="primary"] svg,
        div[data-testid="stButton"] button[kind="primary"] path {
            fill: #000000 !important;
        }
        div[data-testid="stDownloadButton"] button[kind="primary"],
        div[data-testid="stDownloadButton"] button[kind="primary"] p,
        div[data-testid="stDownloadButton"] button[kind="primary"] span {
            color: #000000 !important;
        }

        /* Pastillas Base Web (multiselect, filtros similares): texto negro */
        [data-baseweb="tag"] {
            color: #000000 !important;
        }
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] p {
            color: #000000 !important;
        }
        [data-baseweb="tag"] svg,
        [data-baseweb="tag"] path {
            fill: #000000 !important;
        }

        /* Pestañas principales (st.tabs): etiquetas siempre blancas, clicada o no */
        div[data-testid="stTabs"] [data-baseweb="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] * {
            color: #FFFFFF !important;
        }

        /* Valor numérico junto al mango del slider (color primario) */
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Fitness Coach")
    st.caption("Crea y personaliza tu plan de entrenamiento con IA, utilizando la documentacion cientifica para crear los mejores planes de entrenamiento.")

    # Initialize session state
    if "fitness_coach" not in st.session_state:
        st.session_state.fitness_coach = AIFitnessCoach()

    tab1, tab2 = st.tabs(["Crear plan de entrenamiento", "Actualizar plan"])

    with tab1:
        st.header("Crea tu plan de entrenamiento personalizado")
        st.caption(
            "Completá lo básico primero. Los bloques plegables **mejoran mucho el plan** si los rellenás; priorizá equipamiento y disponibilidad real."
        )

        st.subheader("Datos esenciales")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Edad", min_value=1, max_value=120)
        with col2:
            weight = st.number_input("Peso (kg)", min_value=1.0)
        with col3:
            height = st.number_input("Estatura (cm)", min_value=1.0)

        gender = st.radio("Género", ["Masculino", "Femenino", "Otro"], horizontal=True)

        primary_goal = st.selectbox(
            "Objetivo principal",
            ["Pérdida de peso", "Ganancia muscular", "Mejorar resistencia", "Condición física general"],
        )

        target_timeframe = st.selectbox(
            "Plazo objetivo",
            ["3 meses", "6 meses", "1 año"],
        )

        c_pref1, c_pref2 = st.columns(2)
        with c_pref1:
            workout_preferences = st.multiselect(
                "Tipos de entrenamiento preferidos",
                ["Cardio", "Fuerza / musculación", "Yoga", "Pilates", "Flexibilidad", "HIIT"],
            )
        with c_pref2:
            workout_days = st.multiselect(
                "Días preferidos para entrenar",
                ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
            )

        workout_duration = st.slider(
            "Duración preferida por sesión (minutos)",
            min_value=15,
            max_value=120,
            step=15,
        )

        activity_level = st.radio(
            "Nivel de actividad actual (fuera del gimnasio)",
            ["Sedentario", "Ligeramente activo", "Moderadamente activo", "Muy activo"],
            horizontal=True,
        )

        health_conditions = st.text_area(
            "Otras condiciones de salud relevantes (opcional)",
            placeholder="Ej.: hipertensión controlada, diabetes…",
            height=72,
        )

        # —— Alta prioridad: composición, historial, equipamiento, disponibilidad real ——
        with st.expander("Composición corporal, equipamiento y horarios (alta prioridad)", expanded=False):
            st.markdown("**Composición** — dos personas con el mismo peso pueden necesitar planes distintos.")
            c_bf1, c_bf2, c_bf3 = st.columns(3)
            with c_bf1:
                knows_body_fat = st.checkbox("Conozco mi % de grasa corporal (aprox.)", value=False)
            with c_bf2:
                body_fat_percent = (
                    st.number_input("% Grasa corporal", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
                    if knows_body_fat
                    else None
                )
            with c_bf3:
                st.caption("Si no sabés el %, dejá el checkbox desmarcado.")

            c_w, c_h = st.columns(2)
            with c_w:
                waist_cm = st.number_input("Perímetro de cintura (cm), 0 si no mediste", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
            with c_h:
                hip_cm = st.number_input("Perímetro de cadera (cm), 0 si no mediste", min_value=0.0, max_value=200.0, value=0.0, step=0.5)

            st.divider()
            st.markdown("**Equipamiento disponible** — sin esto el plan puede no aplicarse a tu espacio.")
            equipment_available = st.multiselect(
                "¿Qué tenés a mano?",
                [
                    "Gym comercial completo",
                    "Gimnasio en casa (rack, mancuernas, banco)",
                    "Solo peso corporal",
                    "Bandas elásticas / minibands",
                    "Kettlebell(s) o mancuernas sueltas",
                    "Máquinas limitadas (edificio, hotel)",
                    "Cinta o elíptica en casa",
                ],
            )
            equipment_notes = st.text_area("Aclaraciones de equipamiento (opcional)", height=60)

            st.divider()
            st.markdown("**Disponibilidad real**")
            training_time_of_day = st.selectbox(
                "Horario habitual en que entrenarías",
                [
                    "Mañana (antes de las 9)",
                    "Media mañana",
                    "Mediodía",
                    "Tarde",
                    "Noche",
                    "Variable / según día",
                ],
            )
            hours_per_week_total = st.slider(
                "Horas totales aproximadas disponibles por semana para entrenar",
                min_value=0.5,
                max_value=20.0,
                value=4.0,
                step=0.5,
                help="Además de los días elegidos arriba: cuánto tiempo real podés dedicar en total.",
            )

        with st.expander("Historial de entrenamiento (alta prioridad)", expanded=False):
            time_without_training = st.selectbox(
                "¿Cuánto tiempo llevás sin entrenar con constancia?",
                [
                    "Entreno con constancia ahora",
                    "Menos de 1 mes",
                    "1 a 3 meses",
                    "3 a 6 meses",
                    "Más de 6 meses",
                    "Nunca entrené con constancia",
                ],
            )
            previous_training_types = st.multiselect(
                "¿Qué has hecho antes?",
                [
                    "Pesas / musculación",
                    "Cardio en máquina",
                    "Running / caminatas largas",
                    "Deportes de equipo",
                    "Clases grupales",
                    "Crossfit / funcional",
                    "Yoga / pilates",
                    "Otro",
                ],
            )
            previous_training_notes = st.text_area(
                "Detalle de entrenamientos previos u otros (opcional)",
                height=64,
            )
            why_stopped_training = st.text_area(
                "Si dejaste de entrenar, ¿por qué? (lesión, falta de tiempo, motivación, mudanza…)",
                height=72,
            )

        # —— Impacto medio ——
        with st.expander("Rendimiento base, trabajo, sueño y estrés", expanded=False):
            st.markdown("**Métricas aproximadas** (mejor estimación honesta).")
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                max_pullups = st.number_input("Dominadas seguidas (máx.) — 0 si no podés / no probaste", min_value=0, max_value=80, value=0)
            with c_m2:
                max_pushups = st.number_input("Flexiones seguidas (máx.) — 0 si no probaste", min_value=0, max_value=150, value=0)
            can_run_1km_nonstop = st.radio(
                "¿Podés correr ~1 km sin parar?",
                ["Sí", "No", "No estoy seguro/a"],
                horizontal=True,
            )
            knows_resting_hr = st.checkbox("Conozco mi frecuencia cardíaca en reposo", value=False)
            resting_heart_rate = (
                st.number_input("FC en reposo (latidos/min)", min_value=35, max_value=120, value=65)
                if knows_resting_hr
                else None
            )

            st.divider()
            job_activity_type = st.radio(
                "Trabajo / día a día principalmente",
                ["Sentado / oficina", "De pie o en movimiento", "Mixto"],
                horizontal=True,
            )
            stress_level_1_5 = st.slider("Estrés percibido (1 = bajo, 5 = muy alto)", 1, 5, 3)
            sleep_hours_avg = st.slider("Horas de sueño promedio por noche", 4.0, 12.0, 7.0, step=0.5)

        with st.expander("Lesiones y limitaciones de movimiento", expanded=False):
            injury_status = st.radio(
                "Historial de lesiones",
                ["Sin lesiones relevantes", "Lesión activa (me limita ahora)", "Solo lesiones pasadas (rehabilitadas)"],
            )
            movement_limitations = st.text_area(
                "Limitaciones concretas (ej.: no sentadilla profunda, no press vertical, no impacto…)",
                height=80,
            )

        # —— Refinamiento y nutrición ——
        with st.expander("Objetivos detallados, nutrición y preferencias", expanded=False):
            target_weight_change_kg = st.number_input(
                "Meta de cambio de peso en kg (positivo = subir, negativo = bajar, 0 = no especifico)",
                min_value=-80.0,
                max_value=80.0,
                value=0.0,
                step=0.5,
            )
            goal_event_note = st.text_input(
                "¿Hay fecha o evento objetivo? (opcional, texto libre)",
                placeholder="Ej.: boda en junio, viaje en marzo…",
            )
            goal_focus_priorities = st.multiselect(
                "¿Qué priorizás más? (podés marcar varios)",
                ["Estética / composición corporal", "Salud y longevidad", "Rendimiento deportivo"],
            )

            st.divider()
            trains_solo_or_group = st.radio(
                "¿Cómo preferís entrenar?",
                ["Solo/a", "Acompañado/a o en grupo", "Me da igual"],
                horizontal=True,
            )
            routine_style_preference = st.radio(
                "Estilo de rutina",
                ["Prefiero rutinas fijas y repetibles", "Prefiero mucha variedad", "Me da igual"],
            )
            exercises_disliked = st.text_area("Ejercicios que no tolerás u odiás (adherencia)", height=64)
            past_success_methods = st.text_area("¿Algo que te haya funcionado bien antes? (opcional)", height=64)

            st.divider()
            meals_per_day = st.slider("Comidas aproximadas por día", 1, 8, 3)
            calorie_intention = st.selectbox(
                "Enfoque calórico aproximado (si lo sabés)",
                ["No lo sé / no llevo control", "Déficit (perder grasa)", "Mantenimiento", "Superávit (ganar masa)"],
            )
            supplements = st.text_area("Suplementos (opcional)", height=48)
            dietary_preferences = st.text_area(
                "Preferencias o restricciones alimentarias (texto libre)",
                height=64,
            )

        # —— Motivación y barreras ——
        with st.expander("Motivación y barreras (muy valioso para adherencia)", expanded=False):
            barrier_why_not_before = st.text_area(
                "¿Por qué creés que no lograste el objetivo antes? (honestidad ayuda al plan)",
                height=72,
            )
            barrier_plan_adherence = st.text_area(
                "¿Qué podría impedirte seguir el plan? (viajes, turnos, familia, etc.)",
                height=72,
            )
            motivation_scale_1_10 = st.slider("Motivación en este momento (1–10)", 1, 10, 5)

        if st.button("Generar plan de entrenamiento", type="primary", key="btn_generate_plan"):
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
                "health_conditions": health_conditions or None,
                "dietary_preferences": dietary_preferences or None,
                "body_fat_percent": body_fat_percent,
                "waist_cm": waist_cm if waist_cm > 0 else None,
                "hip_cm": hip_cm if hip_cm > 0 else None,
                "time_without_training": time_without_training,
                "previous_training_types": previous_training_types,
                "previous_training_notes": previous_training_notes or None,
                "why_stopped_training": why_stopped_training or None,
                "equipment_available": equipment_available,
                "equipment_notes": equipment_notes or None,
                "training_time_of_day": training_time_of_day,
                "hours_per_week_total": hours_per_week_total,
                "max_pullups": max_pullups,
                "max_pushups": max_pushups,
                "can_run_1km_nonstop": can_run_1km_nonstop,
                "resting_heart_rate": resting_heart_rate,
                "job_activity_type": job_activity_type,
                "stress_level_1_5": stress_level_1_5,
                "sleep_hours_avg": sleep_hours_avg,
                "injury_status": injury_status,
                "movement_limitations": movement_limitations or None,
                "trains_solo_or_group": trains_solo_or_group,
                "routine_style_preference": routine_style_preference,
                "exercises_disliked": exercises_disliked or None,
                "past_success_methods": past_success_methods or None,
                "meals_per_day": meals_per_day,
                "calorie_intention": calorie_intention,
                "supplements": supplements or None,
                "target_weight_change_kg": target_weight_change_kg if target_weight_change_kg != 0 else None,
                "goal_event_note": goal_event_note or None,
                "goal_focus_priorities": goal_focus_priorities,
                "barrier_why_not_before": barrier_why_not_before or None,
                "barrier_plan_adherence": barrier_plan_adherence or None,
                "motivation_scale_1_10": motivation_scale_1_10,
            }

            with st.spinner("Generando tu plan personalizado..."):
                messages = st.session_state.fitness_coach.run(user_data)
                st.session_state.last_plan = messages

            for message in messages:
                st.write(f"**{_etiqueta_mensaje(message.type)}:** {message.content}")

    with tab2:
        st.header("Actualizar tu plan de entrenamiento")
        feedback = st.text_area("Comentarios sobre tu plan actual:")
        
        if st.button("Actualizar plan", key="btn_update_plan"):
            with st.spinner("Actualizando tu plan..."):
                messages = st.session_state.fitness_coach.run({"feedback": feedback})
                
            for message in messages:
                st.write(f"**{_etiqueta_mensaje(message.type)}:** {message.content}")

if __name__ == "__main__":
    main()
