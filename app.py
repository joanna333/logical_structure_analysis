# app.py
import streamlit as st

def main_page():
    st.title("Welcome to the TMS Preparation Tool!")
    st.write("Select a function from the navigation menu")
    
    # Quick access buttons with unique keys
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Analyze Text", key="main_analyze"):
            st.session_state.page = "analyze"
    with col2:
        if st.button("Generate Questions", key="main_generate"):
            st.session_state.page = "generate"
    with col3:
        if st.button("View Results", key="main_results"):
            st.session_state.page = "results"

def analyze_page():
    st.title("Text Analysis")
    
    # Source selection
    source_type = st.radio(
        "Select Text Source",
        ["Wikipedia Articles", "Custom Text"],
        key="source_type"
    )
    
    if source_type == "Wikipedia Articles":
        col1, col2 = st.columns([2,1])
        with col1:
            # Wikipedia search
            wiki_topic = st.text_input(
                "Enter medical topic to search",
                placeholder="e.g. Cardiovascular system",
                key="wiki_topic"
            )
            
            if st.button("Search Wikipedia", key="wiki_search"):
                if wiki_topic:
                    st.info("Searching Wikipedia...")
                    # TODO: Implement Wikipedia search
        
        with col2:
            # Topic suggestions
            st.subheader("Popular Topics")
            topics = ["Anatomy", "Physiology", "Neurology", "Cell Biology"]
            for topic in topics:
                if st.button(topic, key=f"topic_{topic.lower()}"):
                    st.session_state.wiki_topic = topic
    
    else:  # Custom Text
        st.subheader("Upload or Paste Custom Text")
        upload_tab, paste_tab = st.tabs(["Upload File", "Paste Text"])
        
        with upload_tab:
            st.file_uploader(
                "Upload Text File",
                type=["txt", "docx", "pdf"],
                key="file_upload"
            )
            
        with paste_tab:
            st.text_area(
                "Paste your text here:",
                height=200,
                key="text_input"
            )
    
    # Text Preview and Analysis
    if st.session_state.get("wiki_topic") or st.session_state.get("text_input"):
        st.subheader("Text Preview")
        # Show text preview in expandable section
        with st.expander("Show/Hide Text"):
            st.write("Preview text here...")  # TODO: Show actual text
        
        col1, col2 = st.columns(2)
        with col1:
            st.button("Analyze Logical Structure", key="analyze_button")
        with col2:
            st.button("Generate Questions", key="quick_generate")

def generate_page():
    st.title("Question Generator")
    
    # Sample text for questions
    sample_text = """
    The cardiovascular system plays a crucial role in maintaining homeostasis. Blood pressure regulation 
    involves multiple factors working together. Cardiac output, which is the amount of blood pumped by 
    the heart per minute, directly affects blood pressure. Blood vessels can dilate or constrict to 
    regulate blood flow and pressure. When blood vessels dilate, this increases blood flow to tissues 
    while decreasing overall blood pressure. Various hormones, including angiotensin and vasopressin, 
    help maintain blood pressure within normal ranges by affecting both cardiac output and blood vessel diameter.
    """
    
    # Display source text in expandable section
    with st.expander("Read Text", expanded=True):
        st.markdown(f"**Source Text:**\n{sample_text}")
    
    st.selectbox("Question Type", ["Causal", "Conditional", "Comparative", "All"], key="question_type")
    st.slider("Number of Questions", 1, 5, 3, key="question_count")
    
    # Initialize session state for answers if not exists
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    
    if st.button("Generate Questions", key="generate_button"):
        # Store questions in session state
        if 'questions' not in st.session_state:
            st.session_state.questions = [
                {
                    "question": "Which factors contribute to blood pressure regulation?",
                    "options": [
                        {"text": "Cardiac output", "correct": True},
                        {"text": "Blood vessel diameter", "correct": True},
                        {"text": "Body temperature", "correct": False},
                        {"text": "Hormonal regulation", "correct": True}
                    ],
                    "type": "Causal"
                },
                {
                    "question": "What would happen if the blood vessels dilate?",
                    "options": [
                        {"text": "Blood pressure increases", "correct": False},
                        {"text": "Blood pressure decreases", "correct": True},
                        {"text": "Blood flow increases", "correct": True},
                        {"text": "Heart rate remains constant", "correct": False}
                    ],
                    "type": "Conditional"
                }
            ]
    
    # Display questions if they exist in session state
    if 'questions' in st.session_state:
        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"### Question {i+1} ({q['type']})")
            st.write(q["question"])
            
            # Create columns for each option
            cols = st.columns(len(q["options"]))
            
            # Initialize answers in session state if not exists
            if f"q{i}" not in st.session_state.user_answers:
                st.session_state.user_answers[f"q{i}"] = [False] * len(q["options"])
            
            # Display options as checkboxes
            for j, (opt, col) in enumerate(zip(q["options"], cols)):
                with col:
                    st.session_state.user_answers[f"q{i}"][j] = st.checkbox(
                        opt["text"],
                        value=st.session_state.user_answers[f"q{i}"][j],
                        key=f"q{i}_opt{j}"
                    )
            
            # Check answers button
            if st.button("Check Answers", key=f"check_q{i}"):
                correct_answers = [opt["correct"] for opt in q["options"]]
                user_answers = st.session_state.user_answers[f"q{i}"]
                
                if user_answers == correct_answers:
                    st.success("Correct! Well done!")
                else:
                    st.error("Not quite right. Try again!")
                    # Show hint
                    st.info("Hint: Review the text section about blood pressure regulation.")
            
            st.markdown("---")

def results_page():
    st.title("Question Results")
    
    # Initialize results data from session state or use default values
    if 'questions' in st.session_state and 'user_answers' in st.session_state:
        # Calculate results from session state
        total_questions = len(st.session_state.questions)
        correct_answers = sum(
            1 for i, q in enumerate(st.session_state.questions)
            if st.session_state.user_answers.get(f"q{i}") == 
            [opt["correct"] for opt in q["options"]]
        )
        
        results_data = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "question_results": [
                {
                    "question": q["question"],
                    "user_answers": [
                        opt["text"] for j, opt in enumerate(q["options"])
                        if st.session_state.user_answers.get(f"q{i}")[j]
                    ],
                    "correct_answers": [
                        opt["text"] for opt in q["options"]
                        if opt["correct"]
                    ],
                    "explanation": "Review the logical structure in the text above.",
                    "type": q["type"]
                }
                for i, q in enumerate(st.session_state.questions)
            ]
        }
    else:
        # Default data if no questions attempted
        results_data = {
            "total_questions": 0,
            "correct_answers": 0,
            "question_results": []
        }
    
    # Add text with highlighted logical structure
    st.subheader("Text Analysis")
    
    # First define styles separately
    styles = """
    <style>
        .text-container { 
            background-color: #f5f5f5; 
            padding: 30px; 
            border-radius: 10px; 
            margin: 20px 0;
        }
        .logical-text {
            font-size: 16px;
            line-height: 1.8;
            white-space: pre-line;
        }
        .cause { background-color: #ffcdd2; }
        .effect { background-color: #c8e6c9; }
        .condition { background-color: #fff9c4; }
        .relation { background-color: #bbdefb; }
        .cause, .effect, .condition, .relation {
            padding: 3px 6px;
            border-radius: 4px;
            display: inline-block;
        }
        .legend {
            background: white;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .legend-item {
            display: inline-block;
            margin: 0 10px;
            padding: 5px 10px;
            border-radius: 4px;
        }
    </style>
    """

    # Then define content
    content = f"""
    {styles}
    <div class="legend">
        <span class="legend-item cause">🔍 Cause</span>
        <span class="legend-item effect">✨ Effect</span>
        <span class="legend-item condition">⚡ Condition</span>
        <span class="legend-item relation">🔄 Relation</span>
    </div>
    
    <div class="text-container">
        <div class="logical-text">
            <p>
                <span class="cause">Cardiac output</span>, which is the amount of blood pumped by the heart per minute, 
                <span class="relation">directly affects</span> 
                <span class="effect">blood pressure</span>. <span class="condition">When blood vessels dilate</span>, <span class="effect">this increases blood flow to tissues while decreasing overall blood pressure</span>. <span class="cause">Various hormones, including angiotensin and vasopressin</span>, 
                <span class="relation">help maintain</span> 
                <span class="effect">blood pressure within normal ranges</span> by 
                <span class="relation">affecting</span> both 
                <span class="effect">cardiac output and blood vessel diameter</span>.
            </p>
        </div>
    </div>
    """

    st.markdown(content, unsafe_allow_html=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{results_data['correct_answers']}/{results_data['total_questions']}")
    with col2:
        percentage = (results_data['correct_answers'] / results_data['total_questions'] * 100) if results_data['total_questions'] > 0 else 0
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        st.metric("Questions Reviewed", results_data['total_questions'])

    # Detailed results
    st.subheader("Detailed Analysis")
    
    for i, result in enumerate(results_data["question_results"], 1):
        with st.expander(f"Question {i}: {result['question']}", expanded=True):
            display_answer_analysis(result)

    # Export results
    st.download_button(
        "Export Results (PDF)",
        "results_data",
        file_name="quiz_results.pdf",
        mime="application/pdf",
        key="export_results"
    )

def display_answer_analysis(result):
    col1, col2 = st.columns([2,1])
    
    with col1:
        # User's Answer Analysis
        st.markdown("#### Your Answer")
        for answer in result["user_answers"]:
            st.markdown(f"- {answer}")
        
        # Correct Answer
        st.markdown("#### Correct Answer")
        for answer in result["correct_answers"]:
            st.markdown(f"- {answer}")
        
        # Correctness evaluation
        if set(result["user_answers"]) == set(result["correct_answers"]):
            st.success("✅ Perfect Understanding!")
            st.markdown("You correctly identified all logical connections.")
        elif any(ans in result["correct_answers"] for ans in result["user_answers"]):
            st.warning("⚠️ Partial Understanding")
            missed = set(result["correct_answers"]) - set(result["user_answers"])
            st.markdown("**Missed connections:**")
            for m in missed:
                st.markdown(f"- {m}")
        else:
            st.error("❌ Review Needed")
            st.markdown("**Focus on:**")
            st.markdown("- Identifying cause-effect relationships")
            st.markdown("- Understanding conditional statements")
            st.markdown("- Recognizing logical connections")
    
    with col2:
        # Related Text & Hints
        st.markdown("#### Key Text Segment")
        st.info(result.get("related_text", "Review the highlighted text above"))
        
        # Improvement Tips
        st.markdown("#### Tips")
        if result.get("type"):
            tips = {
                "Causal": "Look for words like 'causes', 'leads to', 'results in'",
                "Conditional": "Focus on 'if-then' relationships",
                "Comparative": "Notice comparisons and contrasts"
            }
            st.markdown(f"_{tips.get(result['type'], '')}_")

def settings_page():
    st.title("Settings")
    st.selectbox("Language", ["English", "German"], key="language")
    st.multiselect("Text Sources", ["Biology", "Medicine", "Physics"], key="sources")
    st.checkbox("Enable GNN Analysis", key="gnn_enabled")

def main():
    st.set_page_config(page_title="TMS Prep Tool", layout="wide")
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "main"
    
    # Navigation sidebar with unique keys
    with st.sidebar:
        st.title("Navigation")
        if st.button("Home", key="nav_home"): 
            st.session_state.page = "main"
        if st.button("Analyze Text", key="nav_analyze"): 
            st.session_state.page = "analyze"
        if st.button("Generate Questions", key="nav_generate"): 
            st.session_state.page = "generate"
        if st.button("Results", key="nav_results"): 
            st.session_state.page = "results"
        if st.button("Settings", key="nav_settings"): 
            st.session_state.page = "settings"
    
    # Page routing
    pages = {
        "main": main_page,
        "analyze": analyze_page,
        "generate": generate_page,
        "results": results_page,
        "settings": settings_page
    }
    
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()