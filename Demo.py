import streamlit as st
import pandas as pd
from inference import inference

st.set_page_config(page_title="Demo", layout='wide')

# st.markdown("""
#     <style>
#     p {
#         font-size: 22px !important;
#     }
#     li {
#         font-size: 22px !important;
#     }
#     </style>
    
# """, unsafe_allow_html=True)

language = st.selectbox(
    "Select Language",
    ("English", "Vietnamese")
)

# Sidebar mục lục
st.sidebar.title("📚 Table of Contents")

st.sidebar.markdown("""
- [Project introduction](#project_introduction)
- [Confusion matrix](#confusion_matrix)
- [Classification report](#classification_report)
- [Results from other backbones](#backbone)
- [Some sample data](#sample_data)
- [Demo model](#demo)
""", unsafe_allow_html=True)

  
if language == 'Vietnamese':  
    st.markdown('<a id="project_introduction"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Giới thiệu project</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/audio_dl.jpg", use_container_width=True)
    st.write("Trong project này, tôi đã trích xuất Mel spectrogram từ audio gốc, sau đó đưa vào CAFormer để lấy ra các feature maps và dùng XGBoost để phân loại sau cùng. Tôi đã train model trên 2 tập dữ liệu là CREMA-D và RAVDESS. Trong đó, những file âm thanh của RAVDESS thể hiện cảm xúc rõ ràng hơn nên kết quả trên RAVDESS cao hơn nhiều so với CREMA-D. Mặc dù XGBoost không cải tiến kết quả quá nhiều, nhưng nó giúp cân bằng model, khiến model không quá chú ý vào việc phân biệt các class dễ mà quên các class khó. Dưới đây là một số kết quả và demo model")
    
    kaggle_dataset = "https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en"
    st.write("You can find this dataset here: [link](%s)" % kaggle_dataset)
    
    st.markdown('<a id="confusion_matrix"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Confusion matrix of CREMA-D</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/normalized_confusion_matrix.png', use_container_width=True)
    with col2:
        st.image('images/normalized_confusion_matrix_xgb.png', use_container_width=True)
    st.markdown('<h3 style="color:#FFFF99;">Confusion matrix of RAVDESS</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/RAVDESS_normalized_model.png', use_container_width=True)
    with col2:
        st.image('images/RAVDESS_normalized_xgb.png', use_container_width=True)
        
    
    # Label mapping (theo thứ tự bạn đã chỉ: ANG DIS FEA HAP NEU SAD)
    labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

    # Report 1
    report1_data = {
        'Label': labels,
        'Precision': [0.79, 0.59, 0.44, 0.89, 0.66, 0.49],
        'Recall':    [0.76, 0.52, 0.67, 0.32, 0.64, 0.63],
        'F1-Score':  [0.78, 0.56, 0.53, 0.47, 0.65, 0.55],
    }

    # Report 2
    report2_data = {
        'Label': labels,
        'Precision': [0.76, 0.59, 0.61, 0.67, 0.67, 0.55],
        'Recall':    [0.82, 0.63, 0.53, 0.64, 0.63, 0.60],
        'F1-Score':  [0.79, 0.61, 0.56, 0.65, 0.65, 0.57],
    }

    df1 = pd.DataFrame(report1_data)
    df2 = pd.DataFrame(report2_data)
    

    st.markdown('<a id="classification_report"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">So sánh Classification Reports on CREMA-D</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CAFormer")
        st.dataframe(df1.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))

    with col2:
        st.markdown("### XGBoost")
        st.dataframe(df2.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))
    
    
    
    # Label mapping (theo thứ tự bạn đã chỉ: ANG DIS FEA HAP NEU SAD)
    ravdess_labels = ['ANGRY', 'CALM', 'DIGUST', 'FEARFUL', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']

    # Report 1
    ravdess_report1_data = {
        'Label': ravdess_labels,
        'Precision': [0.85, 0.74, 0.85, 0.81, 0.64, 0.58, 0.57, 0.78],
        'Recall':    [0.87, 0.74, 0.76, 0.74, 0.69, 0.74, 0.51, 0.82],
        'F1-Score':  [0.86, 0.74, 0.81, 0.77, 0.67, 0.65, 0.54, 0.79],
    }


    # Report 2
    ravdess_report2_data = {
        'Label': ravdess_labels,
        'Precision': [0.79, 0.79, 0.75, 0.76, 0.64, 0.62, 0.61, 0.89],
        'Recall':    [0.79, 0.82, 0.79, 0.67, 0.74, 0.79, 0.49, 0.87],
        'F1-Score':  [0.79, 0.81, 0.77, 0.71, 0.69, 0.70, 0.54, 0.88],
    }

    ravdess_df1 = pd.DataFrame(ravdess_report1_data)
    ravdess_df2 = pd.DataFrame(ravdess_report2_data)
    st.markdown('<h3 style="color:#FFFF99;">So sánh Classification Reports on RAVDESS</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CAFormer")
        st.dataframe(ravdess_df1.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))

    with col2:
        st.markdown("### XGBoost")
        st.dataframe(ravdess_df2.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))


    st.markdown('<a id="backbone"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Kết quả một số backbone khác</h3>', unsafe_allow_html=True)
    data = {
        "ConvFormer": [
            {"Model": "Conv M 22k 1k", "Dropout": 0.3, "Acc1": 64, "Acc2": 66},
            {"Model": "Conv B 1k",     "Dropout": 0.1, "Acc1": 61, "Acc2": 62},
            {"Model": "Conv B 22k",    "Dropout": 0.3, "Acc1": 64, "Acc2": 63},
        ],
        "CAFormer": [
            {"Model": "CA B 22k 1k",   "Dropout": 0.3, "Acc1": 62, "Acc2": 63},
            {"Model": "CA B 22k",      "Dropout": 0.5, "Acc1": 63, "Acc2": 64},
            {"Model": "CA B 22k 1k",   "Dropout": 0.3,  "Acc1": 60, "Acc2": 60},
            {"Model": "CA B 22k 1k",   "Dropout": 0.2, "Acc1": 59, "Acc2": 62},
            {"Model": "CA B 22k 1k",   "Dropout": 0.3, "Acc1": 58, "Acc2": 61},
        ],
        "EfficientNet": [
            {"Model": "EfficientNet B3", "Dropout": 0.3, "Acc1": 57, "Acc2": 58},
        ],
        "ResNet": [
            {"Model": "ResNet 50", "Dropout": 0.3, "Acc1": 54, "Acc2": 54},
            {"Model": "ResNet 50", "Dropout": 0.1, "Acc1": 54, "Acc2": 52},
            {"Model": "ResNet 18", "Dropout": 0.1, "Acc1": 57, "Acc2": 55},
        ]
    }

    st.info("Acc1 is the accuracy of the model and Acc2 is the accuracy of XGBoost when using these models as a backbone to extract feature maps and feed into XGBoost.")

    # Hiển thị từng bảng
    for model_group, entries in data.items():
        st.subheader(f"📌 {model_group}")
        df = pd.DataFrame(entries)
        st.dataframe(df, use_container_width=True)
        
        
    st.markdown('<a id="sample_data"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Một số dữ liệu mẫu CREMA-D</h3>', unsafe_allow_html=True)
    audio_samples = {
        "Angry": "audio/1001_DFA_ANG_XX.wav",
        "Disgust": "audio/1001_DFA_DIS_XX.wav",
        "Happy": "audio/1001_DFA_HAP_XX.wav",
        "Fear": "audio/1001_DFA_FEA_XX.wav",
        "Sadness": "audio/1001_DFA_SAD_XX.wav",
        "Neutral": "audio/1001_DFA_NEU_XX.wav"
    }
    for emotion, audio_path in audio_samples.items():
        cols = st.columns([1, 3])  
        with cols[0]: 
            st.write(emotion)
        with cols[1]:  
            st.audio(audio_path)
    st.markdown('<h3 style="color:#FFFF99;">Một số dữ liệu mẫu RAVDESS</h3>', unsafe_allow_html=True)
    ravdess_samples = {
        "Neutral": "audio/RAVDESS/03-01-01-01-02-01-02.wav",
        "Calm": "audio/RAVDESS/03-01-02-02-01-02-02.wav",
        "Happy": "audio/RAVDESS/03-01-03-02-01-01-02.wav",
        "Sad": "audio/RAVDESS/03-01-04-02-01-01-02.wav",
        "Angry": "audio/RAVDESS/03-01-05-02-02-02-02.wav",
        "Fearful": "audio/RAVDESS/03-01-06-02-02-02-02.wav",
        "Digust": "audio/RAVDESS/03-01-07-02-01-02-02.wav",
        "Surprised": "audio/RAVDESS/03-01-08-02-02-02-02.wav"
    }
    for emotion, audio_path in ravdess_samples.items():
        cols = st.columns([1, 3])  
        with cols[0]: 
            st.write(emotion)
        with cols[1]:  
            st.audio(audio_path)
        
        
    st.markdown('<a id="demo"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Demo model</h3>', unsafe_allow_html=True)
    st.warning("I saved the file on Google Drive, so it will take a little time to download.")
    new_audio = st.audio_input("Ghi âm")
    if new_audio:
        st.audio(new_audio)
        
        model = st.selectbox(
            "Chọn mô hình để dự đoán:",
            ("CREMA-D", "RAVDESS")  
        )

        if model:
            st.write(f"Using model trained on {model} to predict emotions...")
            probs, label = inference(new_audio, model)
            # st.write(f"probs: {probs}, label: {label}")
            if probs:
                st.write(f"Dự đoán: **{label}**")
                st.write("Xác suất:")
                st.json(probs)
                
    audio_uploaded = st.file_uploader("🎤 Tải file audio (wav, mp3,...)", type=["wav", "mp3", "ogg"])
    if audio_uploaded:
        st.audio(audio_uploaded)
        
        
        model = 'RAVDESS'

        if model:
            st.write(f"Using model trained on {model} to predict emotions...")
            probs, label = inference(audio_uploaded, model)
            # st.write(f"probs: {probs}, label: {label}")
            if probs:
                st.write(f"Dự đoán: **{label}**")
                st.write("Xác suất:")
                st.json(probs)
            
else:
    st.markdown('<a id="project_introduction"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Project Introduction</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/audio_dl.jpg", use_container_width=True)

    st.write("In this project, I extracted Mel spectrograms from the original audio, then passed them through CAFormer to obtain feature maps, which were finally classified using XGBoost. I trained the model on two datasets: CREMA-D and RAVDESS. Among these, RAVDESS audio files express emotions more clearly, so the results on RAVDESS were much better than those on CREMA-D. Although XGBoost didn’t significantly improve the results, it helped balance the model by preventing it from focusing too much on easy classes while ignoring harder ones. Below are some results and a model demo.")

    kaggle_dataset = "https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en"
    st.write("You can find this dataset here: [link](%s)" % kaggle_dataset)


    st.markdown('<a id="confusion_matrix"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Confusion matrix of CREMA-D</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/normalized_confusion_matrix.png', use_container_width=True)
    with col2:
        st.image('images/normalized_confusion_matrix_xgb.png', use_container_width=True)

    st.markdown('<h3 style="color:#FFFF99;">Confusion matrix of RAVDESS</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image('images/RAVDESS_normalized_model.png', use_container_width=True)
    with col2:
        st.image('images/RAVDESS_normalized_xgb.png', use_container_width=True)

    # Label mapping (in the specified order: ANG DIS FEA HAP NEU SAD)
    labels = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

    # Report 1
    report1_data = {
        'Label': labels,
        'Precision': [0.79, 0.59, 0.44, 0.89, 0.66, 0.49],
        'Recall':    [0.76, 0.52, 0.67, 0.32, 0.64, 0.63],
        'F1-Score':  [0.78, 0.56, 0.53, 0.47, 0.65, 0.55],
    }

    # Report 2
    report2_data = {
        'Label': labels,
        'Precision': [0.76, 0.59, 0.61, 0.67, 0.67, 0.55],
        'Recall':    [0.82, 0.63, 0.53, 0.64, 0.63, 0.60],
        'F1-Score':  [0.79, 0.61, 0.56, 0.65, 0.65, 0.57],
    }

    df1 = pd.DataFrame(report1_data)
    df2 = pd.DataFrame(report2_data)
    

    st.markdown('<a id="classification_report"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Comparison of Classification Reports on CREMA-D</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CAFormer")
        st.dataframe(df1.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))

    with col2:
        st.markdown("### XGBoost")
        st.dataframe(df2.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))


    # Label mapping (in your specified order: ANG DIS FEA HAP NEU SAD)
    ravdess_labels = ['ANGRY', 'CALM', 'DIGUST', 'FEARFUL', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']

    # Report 1
    ravdess_report1_data = {
        'Label': ravdess_labels,
        'Precision': [0.85, 0.74, 0.85, 0.81, 0.64, 0.58, 0.57, 0.78],
        'Recall':    [0.87, 0.74, 0.76, 0.74, 0.69, 0.74, 0.51, 0.82],
        'F1-Score':  [0.86, 0.74, 0.81, 0.77, 0.67, 0.65, 0.54, 0.79],
    }

    # Report 2
    ravdess_report2_data = {
        'Label': ravdess_labels,
        'Precision': [0.79, 0.79, 0.75, 0.76, 0.64, 0.62, 0.61, 0.89],
        'Recall':    [0.79, 0.82, 0.79, 0.67, 0.74, 0.79, 0.49, 0.87],
        'F1-Score':  [0.79, 0.81, 0.77, 0.71, 0.69, 0.70, 0.54, 0.88],
    }

    ravdess_df1 = pd.DataFrame(ravdess_report1_data)
    ravdess_df2 = pd.DataFrame(ravdess_report2_data)
    st.markdown('<h3 style="color:#FFFF99;">Comparison of Classification Reports on RAVDESS</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CAFormer")
        st.dataframe(ravdess_df1.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))

    with col2:
        st.markdown("### XGBoost")
        st.dataframe(ravdess_df2.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'}))



    st.markdown('<a id="backbone"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Results from other backbones</h3>', unsafe_allow_html=True)
    data = {
        "ConvFormer": [
            {"Model": "Conv M 22k 1k", "Dropout": 0.3, "Acc1": 64, "Acc2": 66},
            {"Model": "Conv B 1k",     "Dropout": 0.1, "Acc1": 61, "Acc2": 62},
            {"Model": "Conv B 22k",    "Dropout": 0.3, "Acc1": 64, "Acc2": 63},
        ],
        "CAFormer": [
            {"Model": "CA B 22k 1k",   "Dropout": 0.3, "Acc1": 62, "Acc2": 63},
            {"Model": "CA B 22k",      "Dropout": 0.5, "Acc1": 63, "Acc2": 64},
            {"Model": "CA B 22k 1k",   "Dropout": 0.3, "Acc1": 60, "Acc2": 60},
            {"Model": "CA B 22k 1k",   "Dropout": 0.2, "Acc1": 59, "Acc2": 62},
            {"Model": "CA B 22k 1k",   "Dropout": 0.3, "Acc1": 58, "Acc2": 61},
        ],
        "EfficientNet": [
            {"Model": "EfficientNet B3", "Dropout": 0.3, "Acc1": 57, "Acc2": 58},
        ],
        "ResNet": [
            {"Model": "ResNet 50", "Dropout": 0.3, "Acc1": 54, "Acc2": 54},
            {"Model": "ResNet 50", "Dropout": 0.1, "Acc1": 54, "Acc2": 52},
            {"Model": "ResNet 18", "Dropout": 0.1, "Acc1": 57, "Acc2": 55},
        ]
    }

    st.info("Acc1 is the accuracy of the model and Acc2 is the accuracy of XGBoost when using these models as a backbone to extract feature maps and feed into XGBoost.")

    # Display each table
    for model_group, entries in data.items():
        st.subheader(f"📌 {model_group}")
        df = pd.DataFrame(entries)
        st.dataframe(df, use_container_width=True)


    st.markdown('<a id="sample_data"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Some sample data from CREMA-D</h3>', unsafe_allow_html=True)
    audio_samples = {
        "Angry": "audio/1001_DFA_ANG_XX.wav",
        "Disgust": "audio/1001_DFA_DIS_XX.wav",
        "Happy": "audio/1001_DFA_HAP_XX.wav",
        "Fear": "audio/1001_DFA_FEA_XX.wav",
        "Sadness": "audio/1001_DFA_SAD_XX.wav",
        "Neutral": "audio/1001_DFA_NEU_XX.wav"
    }
    for emotion, audio_path in audio_samples.items():
        cols = st.columns([1, 3])
        with cols[0]:
            st.write(emotion)
        with cols[1]:
            st.audio(audio_path)

    st.markdown('<h3 style="color:#FFFF99;">Some sample data from RAVDESS</h3>', unsafe_allow_html=True)
    ravdess_samples = {
        "Neutral": "audio/RAVDESS/03-01-01-01-02-01-02.wav",
        "Calm": "audio/RAVDESS/03-01-02-02-01-02-02.wav",
        "Happy": "audio/RAVDESS/03-01-03-02-01-01-02.wav",
        "Sad": "audio/RAVDESS/03-01-04-02-01-01-02.wav",
        "Angry": "audio/RAVDESS/03-01-05-02-02-02-02.wav",
        "Fearful": "audio/RAVDESS/03-01-06-02-02-02-02.wav",
        "Digust": "audio/RAVDESS/03-01-07-02-01-02-02.wav",
        "Surprised": "audio/RAVDESS/03-01-08-02-02-02-02.wav"
    }
    for emotion, audio_path in ravdess_samples.items():
        cols = st.columns([1, 3])  
        with cols[0]: 
            st.write(emotion)
        with cols[1]:  
            st.audio(audio_path)
        
        
    st.markdown('<a id="demo"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#FFFF99;">Demo model</h3>', unsafe_allow_html=True)
    st.warning("I saved the file on Google Drive, so it will take a little time to download.")
    new_audio = st.audio_input("Ghi âm")
    if new_audio:
        st.audio(new_audio)
        
        model = st.selectbox(
            "Choose model to use:",
            ("CREMA-D", "RAVDESS")  
        )

        if model:
            st.write(f"Using model trained on {model} to predict emotions...")
            probs, label = inference(new_audio, model)
            # st.write(f"probs: {probs}, label: {label}")
            if probs:
                st.write(f"Predicted emotion: **{label}**")
                st.write("Probability:")
                st.json(probs)
                
    audio_uploaded = st.file_uploader("🎤 Upload file audio (wav, mp3,...)", type=["wav", "mp3", "ogg"])
    if audio_uploaded:
        st.audio(audio_uploaded)
        
        
        model = st.selectbox(
            "Choose model to use:",
            ("CREMA-D", "RAVDESS")  
        )

        if model:
            st.write(f"Using model trained on {model} to predict emotions...")
            probs, label = inference(audio_uploaded, model)
            # st.write(f"probs: {probs}, label: {label}")
            if probs:
                st.write(f"Predicted emotion - {model}: **{label}**")
                st.write(f"Probability - {model}:")
                st.json(probs)