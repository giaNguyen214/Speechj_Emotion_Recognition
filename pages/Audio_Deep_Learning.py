import streamlit as st


st.set_page_config(page_title="Audio deep learning")
 


language = st.selectbox(
    "Select Language",
    ("English", "Vietnamese")
)

st.sidebar.title("📚 Table of Contents")
st.sidebar.markdown("""
- [Some important hyperparameters](#hyperparam)
- [Basic knowledge when doing with audio](#knowledge)
- [Spectrum (Fourier transform)](#spectrum)
- [Spectrogram](#spectrogram)
- [Mel filterbank](#filterbank)
""", unsafe_allow_html=True)


if language == "Vietnamese":
    st.image("images/audio_dl.jpg")
    st.write("Trong project này, trích xuất Mel spectrogram từ audio gốc, đưa vào Convformer để lấy feature map và dùng XGBoost để phân loại sau cùng")

    st.markdown('<a id="hyperparam"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Một số hyperparameters khi chuyển đổi audio sang Mel spectrogram:</h3>', unsafe_allow_html=True)
    st.markdown("""
        - **sr (sampling rate)**: số mẫu mỗi giây  
        - **n_fft**: số lượng điểm dùng trong Fast Fourier Transform  
        - **hop_length**: khoảng thời gian trượt giữa các đoạn  
        - **n_mel**: số lượng Mel bands  
        - **fmin**: tần số thấp nhất  
        - **fmax**: tần số cao nhất  
        """)


    st.markdown('<a id="knowledge"></a>', unsafe_allow_html=True)
    st.write('<h3 style="color:#1E90FF;">Một số khái niệm cơ bản</h3>', unsafe_allow_html=True)
    st.write("- **Dao động**: một dao động là một lần di chuyển từ vị trí cân bằng rồi quay trở lại vị trí cân bằng. Tương đương một lần lặp lại của sóng.")
    st.write("- **Biên độ (Amplitude)**: mức độ dao động (rộng hay hẹp) của sóng âm. Quyết định âm to hay nhỏ.")
    st.write("- **Tần số (Frequency)**: số lần dao động trong một giây. Quyết định âm cao hay thấp.")
    st.write("- **Âm thanh đơn tần số**: âm chỉ có 1 tần số duy nhất.  \
            Ví dụ: một âm 25Hz kéo dài 5 giây thì luôn luôn là 25 dao động mỗi giây trong suốt 5 giây.")
    st.write("- **Âm thanh tổ hợp (non-single frequency sound)**: là sự kết hợp của nhiều âm thanh đơn tần số.  \
            Ví dụ: âm thanh dài 5 giây thì có thể 2s đầu là 25Hz, 2s tiếp theo là tổ hợp 7Hz và 19Hz, đoạn sau là 13Hz.")
    st.write("- **Âm thanh thực tế**: là sự tổng hợp của nhiều âm đơn tần số.  \
            Để phân tích, ta dùng Fourier Transform để phân rã (decompose) thành các tần số cấu thành nên nó.")
    st.write("- **Analog signal**: tín hiệu liên tục theo thời gian.")
    st.write("- **Digital signal**: tín hiệu rời rạc, lấy mẫu (sampling) từ analog signal.")
    st.image("images/analog_digital.jpg")

        

    st.markdown('<a id="spectrum"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Spectrum (Fourier Transform)</h3>', unsafe_allow_html=True)
    st.image("images/time_Freq_domain.jpg")
    st.write("- **Miền thời gian**: trục x là thời gian, trục y là biên độ.")
    st.write("- **Miền tần số**: trục x là tần số, trục y là biên độ (mức độ xuất hiện tần số đó).")
    st.write("- **Fourier Transform**: giúp chuyển đổi tín hiệu từ miền thời gian sang miền tần số.  \
            Ví dụ: trong một bản hòa nhạc gồm nhiều âm thanh khác nhau cùng vang lên, con người vẫn phân biệt được giọng ca sĩ, tiếng guitar, tiếng piano,...")
    # st.latex(r'''
    #         X(f) = \int_{-\infty}^{\infty} x(t) \cdot e^{-j 2\pi f t} \, dt
    #     ''')
    # st.write("Máy tính không thể xử lý tín hiệu liên tục (analog signal) nên cần chuyển thành tín hiệu rời rạc (digital signal): [0.1, 0.5, 0.9, 0.7, 0.0, -0.6, -0.9, ...]\
    #         Mỗi giá trị là biên độ tại một thời điểm. Máy tính cũng không thể tính tích phân vô hạn, nên dùng công thức Discrete fourier transform:")
    # st.latex(r'''
    #         X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}
    #     ''')
    # st.markdown("Tính theo DFT thì rất chậm $O(N^2)$ nên chuyển qua Fast fourier transform $O(NlogN)$")
    st.write("Kết quả Fourier Transform là **spectrum**, liệt kê các tần số xuất hiện trong âm thanh.  \
            Tuy nhiên **không cho biết thời điểm** xuất hiện các tần số đó. VD một âm thanh có 2 giây đầu là tổ hợp của 2 và 3Hz, 3 giây sau là tổ hợp của 17 và 23Hz.\
                Nhưng fourier transform chỉ cho ta biết âm thanh được tổ hợp từ 2, 3, 17, 23 (Hz) mà không biết thời gian xuất hiện")



    st.markdown('<a id="spectrogram"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Spectrogram</h3>', unsafe_allow_html=True)
    st.markdown(""" 
            **Short-Time Fourier Transform (STFT)** giải quyết việc mất thông tin thời gian bằng cách: 
            
            - Chia tín hiệu audio thành các đoạn nhỏ  
            - Áp dụng Fourier Transform lên từng đoạn -> ra được spectrum 
            - Ghép các kết quả (spectrum) lại theo trục thời gian 
        """)
    st.write("Sau khi tính ra, kết quả có dạng (Time, Frequency, Amplitude) với 2 trục thời gian và tần số, còn biên độ được biểu diễn bằng màu sắc của bức hình\
            Và ta tiếp tục lấy log_scale = log(frequency) và decibel = log(amplitude) => kết quả sẽ có được spectrogram")


    st.markdown('<a id="filterbank"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Mel Filter Bank và Mel Scale</h3>', unsafe_allow_html=True)
    st.markdown("""
            - Khả năng phân biệt tần số của con người **không tuyến tính**.  
            Ví dụ:  
                - 100Hz vs 200Hz: chúng ta có thể nhận ra sự khác nhau của 2 âm thanh -> dễ phân biệt  
                - 4000Hz vs 4100Hz: hầu như không cảm nhận được sự khác biệt -> khó phân biệt mặc dù cũng cách nhau 100Hz

            - Vì vậy, người ta dùng mel filter bank, mục đích là giảm số chiều của trục tần số và chuyển về thang **Mel Scale** - một thang tần số phi tuyến tính để mô phỏng cách tai người cảm nhận âm thanh.
        """)
    st.markdown("""
            - n_mel: Số lượng bộ lọc chia trên trục tần số ở thang Mel. Cũng chính là số chiều của trục tần số sau khi thực hiện. \
                VD: đầu vào là spectrogram (Time=5, Frequency=10000) có 5 frames thời gian, mỗi frame có 10000 giá trị tần số. Thì đầu ra sẽ là (Time=5, Frequency=n_mel)
            - fmin: là điểm bắt đầu của mel filter bank. Nếu lấy cao thì sẽ bỏ qua một số âm, lấy thấp thì nắm bắt được các âm trầm
            - fmax: là điểm kết thúc của mel filter bank. Nếu lấy cao thì nắm bắt được các âm cao, lấy thấp thì bỏ qua một số âm
                    """)
        
            
    st.success("=> Kết quả cuối cùng được biểu diễn dưới dạng ảnh và xử lý như một bài toán image classfication bình thường")
    st.image("images/mel_spectrogram.jpg", caption="Mel spectrogram as a image for downsampling task (RAVDESS)")
    st.image("images/mel_spectrogram_ravdess.jpg", caption="Mel spectrogram as a image for downsampling task (CREMA-D)")
    
else:
    st.image("images/audio_dl.jpg", caption="General pipeline when handling audio input")
    st.markdown('<a id="hyperparam"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Some important hyperparameters when converting audio to Mel spectrogram:</h3>', unsafe_allow_html=True)
    st.markdown("""
        - **sr (sampling rate)**: number of samples per second  
        - **n_fft**: number of points used in the Fast Fourier Transform  
        - **hop_length**: stride (sliding step) between segments  
        - **n_mel**: number of Mel bands  
        - **fmin**: minimum frequency  
        - **fmax**: maximum frequency  
        """)

    st.markdown('<a id="knowledge"></a>', unsafe_allow_html=True)
    st.write('<h3 style="color:#1E90FF;">Some basic concepts</h3>', unsafe_allow_html=True)
    st.write("- **Vibration**: a repeated back-and-forth movement around a central point. It represents one full wave cycle.")
    st.write("- **Amplitude**: how strong or big the vibration is. Larger amplitude means the sound is louder, and smaller amplitude means it's quieter.")
    st.write("- **Frequency**: number of vibrations per second. Determines the pitch (high or low) of the sound.")
    st.write("- **Single-frequency sound**: sound that contains only one frequency.  \
            Example: a 25Hz tone lasting 5 seconds will always vibrate at 25 times per second.")
    st.write("- **Non-single frequency sound**: a combination of multiple single-frequency sounds.  \
            Example: a 5-second sound could consist of 25Hz in the first 2 seconds, a mix of 7Hz and 19Hz in the next 2 seconds, and 13Hz at the end.")
    st.write("- **Real-world sound**: a mixture of many single-frequency sounds.  \
            To analyze it, we use Fourier Transform to decompose it into its frequency components.")
    st.write("- **Analog signal**: a continuous signal over time.")
    st.write("- **Digital signal**: a discrete signal obtained by sampling from the analog signal.")
    st.image("images/analog_digital.jpg", caption="The curve represents the analog signal, while the individual points illustrate the sampling process used to convert it into a digital signal.")

    st.markdown('<a id="spectrum"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Spectrum (Fourier Transform)</h3>', unsafe_allow_html=True)
    st.image("images/time_Freq_domain.jpg")
    st.write("- **Time domain**: x-axis is time, y-axis is amplitude.")
    st.write("- **Frequency domain**: x-axis is frequency, y-axis is amplitude (indicating presence of frequency).")
    st.write("- **Fourier Transform**: converts a signal from time domain to frequency domain.  \
            For example: in a music performance with multiple instruments, humans can still distinguish the singer's voice, the guitar, the piano, etc.")
    # st.latex(r'''
    #         X(f) = \int_{-\infty}^{\infty} x(t) \cdot e^{-j 2\pi f t} \, dt
    #     ''')
    # st.write("Computers cannot process continuous signals (analog), so we convert them to digital signals: [0.1, 0.5, 0.9, 0.7, 0.0, -0.6, -0.9, ...] \
    #         where each value is the amplitude at a time step. Since computers also can’t calculate infinite integrals, we use Discrete Fourier Transform:")
    # st.latex(r'''
    #         X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}
    #     ''')
    # st.markdown("DFT is slow at $O(N^2)$ so we use Fast Fourier Transform with $O(NlogN)$ complexity.")
    st.write("The result of Fourier Transform is the **spectrum**, which lists the frequencies present in the sound.  \
            However, it does **not provide time information**. For example, a sound may consist of 2Hz and 3Hz in the first 2 seconds, then 17Hz and 23Hz in the next 3 seconds. \
                But the Fourier Transform only tells us the frequencies 2, 3, 17, 23 are present, not when.")

    st.markdown('<a id="spectrogram"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Spectrogram</h3>', unsafe_allow_html=True)
    st.markdown(""" 
            **Short-Time Fourier Transform (STFT)** solves the loss of time information by:  
            
            - Splitting the audio signal into small segments  
            - Applying Fourier Transform to each segment → get spectrum  
            - Combining the results along the time axis  
        """)
    st.write("The result has the form (Time, Frequency, Amplitude). Time and frequency are axes, and amplitude is represented by the color of the image. \
            We then apply log scaling: log(frequency) and log(amplitude) → the final result is a spectrogram.")

    st.markdown('<a id="filterbank"></a>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#1E90FF;">Mel Filter Bank and Mel Scale</h3>', unsafe_allow_html=True)
    st.markdown("""
            - Human frequency perception is **non-linear**.  
            For example:  
                - 100Hz vs 200Hz: we can clearly hear the difference  
                - 4000Hz vs 4100Hz: we can hardly tell the difference  

            - Therefore, huamn define something called a mel filter bank to reduce the frequency axis dimensions and convert to **Mel Scale** – a non-linear frequency scale that mimics human auditory perception.
        """)
    st.markdown("""
            - n_mel: Number of filters on the Mel scale. This also becomes the new dimension of the frequency axis. \
                Example: original spectrogram has shape (Time=5, Frequency=10000), after applying filter bank → (Time=5, Frequency=n_mel)
            - fmin: starting frequency of the mel filter bank. A higher fmin will skip some low-frequency sounds; a lower fmin will capture them.
            - fmax: ending frequency of the mel filter bank. A higher fmax captures high-frequency sounds; a lower fmax ignores them.
        """)

    st.success("=> The final result is visualized as an image and processed like a typical image classification task.")
    st.image("images/mel_spectrogram.jpg", caption="Mel spectrogram as an image for downsampling task (RAVDESS)")
    st.image("images/mel_spectrogram_ravdess.jpg", caption="Mel spectrogram as an image for downsampling task (CREMA-D)")
