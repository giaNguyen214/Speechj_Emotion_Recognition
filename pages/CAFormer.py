import streamlit as st

st.set_page_config(page_title="CAFormer", layout='wide')

language = st.selectbox(
    "Select Language",
    ("English", "Vietnamese")
)

# Sidebar mục lục
st.sidebar.title("📚 Table of Contents")

st.sidebar.markdown("""
- [Transformer](#transformer)
- [Attention Mechanism](#attention-mechanism)
- [MetaFormer](#metaformer)
""", unsafe_allow_html=True)


if language == 'Vietnamese':
    st.markdown('<a id="transformer"></a>', unsafe_allow_html=True)
    st.title("Transformer trong xử lý ngôn ngữ tự nhiên")
    st.write("Transformer thực ra là đưa vào một câu và dự đoán token kế tiếp của câu đó. Thật ra, một token không hẳn là một word, nhưng để tiện, chúng ta sử dụng 1 token ~ 1 word")
    st.write("Máy tính chỉ có thể hiểu được những con số mà không hiểu được text mà chúng ta nhập vào. Vì vậy, chúng ta cần embed word thành những vector. \
        Bản chất thì mỗi từ vựng cũng chỉ là cách chúng ta gán ghép quy luật cho vị trí đặt từ và ngữ nghĩa đi kèm. Nếu thay vì ghi 'h' là 'h' thì ta có thể đặt 'h' là 1, tương tự, 'e' 2, 'l' 3, 'o' 4 thì 'hello' sẽ là 12334 \
        Nhưng vì số lượng từ vựng và ngữ nghĩa nhiều, nên không thể biểu diễn dưới dạng những con số đơn giản như vậy được mà cần đến những high-dimension vectors -> Đó là lý do ra đời của Embedding") 
    st.write("Lưu ý là trong GPT, họ tách ra thành từng bộ với chức năng riêng như Token Embedding, Positional Embedding, Querry, Key, Value,...")
 
    st.write("Sau khi đã train và có được một model Embedding, chúng ta đưa các tokens vào và lấy ra được embeddings tương ứng. Tiếp đó, chúng ta cần Position embedding, nếu không thì 'I love you' và 'You love I' sẽ là chung 1 vector set. Thêm vào đó, bởi vì Embedding model được huấn luyện chung nên là các embeddings đầu ra sẽ không có thông tin ngữ cảnh của câu. \
    Ví dụ: The football match was exciting. (match: a sport event) and He lit the fire with a match. (match: a small stick made of wood or cardboard that is used for lighting a fire, cigarette, etc.). Cả 2 từ match đều cùng là noun, nhưng ý nghĩa khác nhau rõ rệt. Mặc dù vậy, khi đưa qua Embedding model thì nó cho ra cùng một kết quả. Điều này là không đúng")

    st.write("Vì vậy, ý nghĩa của Transformer là điều chỉnh các embeddings đó theo context để tạo nên một embedding khác giàu thông tin ngữ cảnh hơn => Attention mechanism")
    st.info("Position embedding là cho biết vị trí của token trong câu, còn Attention là để biết quan hệ giữa các tokens.")


    st.markdown('<a id="attention-mechanism"></a>', unsafe_allow_html=True)
    st.title("Attention mechanism")
    st.write("Để đưa thông tin của các embeddings của các tokens khác vào trong, có một vấn đề cần quan tâm chính là trọng số là bao nhiêu. Nghĩa là embedding khác ảnh hưởng đến target embedding với mức độ nào. Bởi vì có những từ rất quan trọng để xác định context information, nhưng có một số từ không liên quan.")
    st.write("Để tính các trọng số này, chúng ta dùng đến Querry, Key và Value. Ví dụ, ở một token, chúng ta cần nhìn xem xung quanh token đó đang có những từ ngữ gì để bổ sung ngữ nghĩa cho chính bản thân token hiện đang xét. Ví dụ nếu đó là danh từ thì có thể sẽ tìm kiếm các tính từ. Nhưng máy tính không thể hiểu được thế như con người chúng ta, nên cần một model để tạo ra câu hỏi đại loại như kiểu: What information do I need to request from other tokens?. Vai trò của Querry là sinh ra câu hỏi cho chúng ta.")
    st.write("Tương tự, Key chính là câu trả lời của Querry. Có thể hiểu Key như một bản mô tả ngắn về thông tin của token, token giới thiệu mình đang có những gì (không đưa thông tin, nội dung cụ thể của token). Sau đó, tính dot product giữa querry và key và đưa vào softmax để lấy ra kết quả cuối cùng, đây chính là attention scores cho tất cả các tokens khác đối với target token. Chúng thể hiện mức độ đóng góp của các tokens đó vào token hiện tại")
    st.write("Tuy nhiên, nội dung thực sự mà chúng ta cần lấy từ các tokens không phải là embedding gốc ban đầu của chúng mà là Value (một bước nhúng embedding gốc để sinh ra embedding mới chứa gọn giá trị của token)")
    st.write("Khi mà chúng ta tính Querry-Key thì thực chất nó đã bao gồm token hiện tại vào rồi và thường thì trọng số (attention scores) của nó sẽ lớn hơn so với các tokens khác.")
    st.write("Thêm nữa là chúng ta cần masking. Vì bài toán là dự đoán từ tiếp theo dựa trên thông tin câu đã được cho trước, nghĩa là chúng ta sẽ không biết được đằng sau token này sẽ có những gì cho nên chỉ tổng hợp thông tin của các tokens trước đó vào target token thôi")
    st.image("images/querry-key.jpg")
    st.info("Những gì ở trên đây là self-attention. Còn cross-attention là 2 loại data sets khác nhau. Ví dụ như giữa tiếng Anh và tiếng Việt, giữa audio waveform và transcript của nó. Điểm khác biệt ở cross attention là tính querry ở dataset này và key ở dataset kia, còn lại làm như self-attention")
    st.info("Multi-headed attention là dùng nhiều bộ Q, K, V khác nhau để đưa vào các single head self-attention đã trình bày để đưa ra các output khác nhau, sau đó nối các outputs lại và có thể đưa qua một linear layer để gộp thông tin và đưa về đúng số chiều mong muốn")

    col1, col2 = st.columns(2)
    with col1:
        st.image('images/attention.jpg')
    with col2:
        st.image('images/mlp.jpg')
    st.write("Nhìn vào đây chúng ta có thể thấy sự khác biệt rõ ràng giữa Attention và MLP truyền thống. Ở Attention layer là các embedding inputs trao đổi thông tin với nhau tạo ra các embedding outputs. Output của một token phụ thuộc vào nhiều tokens khác. Còn ở MLP thì embedding output của một token chỉ phụ thuộc vào duy nhất embedding input của token đó")


    st.markdown('<a id="metaformer"></a>', unsafe_allow_html=True)
    st.title("MetaFormer for Visual task")

    st.write("**MetaFormer** là kiến trúc chung cho cả thị giác máy tính và xử lý ngôn ngữ tự nhiên, với hai thành phần tách biệt: **Token Mixer** (xử lý tương tác giữa các vị trí) và **Channel MLP** (xử lý tương tác giữa các kênh).")

    # Liệt kê biến thể
    st.write("""
    **Bốn biến thể chính**:
    1. **IdentityFormer** – không có mixer, baseline.  
    2. **RandFormer** – mixer ngẫu nhiên.  
    3. **ConvFormer** – sử dụng convolution làm mixer cục bộ.  
    4. **CAFormer** – sử dụng attention làm mixer toàn cục.  
    Trong đó, IdentityFormer và RandFormer chỉ là 2 model đơn giản để chứng minh tính khả thi của kiến trúc.
    """)

    # Hình ảnh
    _, c2, _ = st.columns([3,1,3])
    with c2:
        st.image("images/metaformer.jpg", caption='Cấu trúc chung của MetaFormer')
    _, c2, _ = st.columns([1,3,1])
    with c2:
        st.image("images/metaformer_benchmark.jpg", caption='MetaFormer benchmark')
    # Pseudo-code block
    st.markdown("**Công thức của một block MetaFormer**:")
    st.code("""
    # I: input
    X     = InputEmbedding(I)
    X'    = X + TokenMixer( Norm1(X) )  # residual + token mixing
    X''   = X' + MLP_ch( Norm2(X') )    # residual + channel MLP
    """, language="python")

    # So sánh CNN và Transformer
    st.write("""
    - **CNN**: là **local mixer**, trộn thông tin trong neighborhood pixel.  
    - **Transformer**: là **global mixer**, attention tổng hợp ngữ cảnh từ mọi token.  
    """)

    # Shape issue
    st.info(
        "MetaFormer xử lý việc bất đồng bộ shape giữa các stage, mixer khác nhau bằng cách đơn giản là **reshape** dữ liệu giữa các stage: "
        "từ H x W x C (CNN) → (H·W) x C (Attention) hoặc ngược lại tùy vào Token Mixer"
    )
    st.write("MetaFormer chia thành 4 giai đoạn. Với input đầu vào là ảnh với shape H x W x 3. Và các kết quả ở stage tiếp theo:")
    _, c2, _ = st.columns(3)    
    with c2:
        st.image("images/stages.jpg")

    st.title("ConvFormer")
    st.write("Ở model này, tác giả sử dụng hoàn toàn là CNN-based block, trong đó mỗi block được thiết kế theo kiến trúc MetaFormer như đã trình bày ở trên, và token mixer họ sử dụng là convolution.")
    _, c2, _ = st.columns(3)
    with c2:
        st.image("images/convformer.jpg", caption="Overall ConvFormer framework")
    _, c2, _ = st.columns([3, 1, 3])
    with c2:
        st.image("images/convformer_block.jpg", caption="ConvFormer block")
        
    with st.expander("Depthwise separable convolution"):
        st.write("Đây là cách hoạt động của CNN bình thường. Chúng ta sẽ có nhiều filter (trong hình dưới đây là 8), và lần lượt đưa input qua từng filter để ra kết quả. Vấn đề là nếu kích thước input lớn thì filter size lớn theo (filter size hiện tại là 8x3x3), dẫn đến tiêu tốn tài nguyên và thời gian tính toán")
        _, c2, _ = st.columns([1, 3, 1])
        with c2:
            st.image("images/normal_cnn.jpg", caption="Normal CNN")
        
        st.write("Do đó, thay vì chúng ta dùng nhiều filter size lớn (8x3x3), mỗi filter duyệt trên toàn bộ channels của input, chúng ta tách riêng thành các filter có kích thước nhỏ lại (4x3x3) và chỉ duyệt trên 1 channel của input. Tuy số lượng filter giữ nguyên nhưng kích thước và số lượng tham số giảm đi đáng kể.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/depthwise.jpg", caption="Depthwise convolution")
        
        st.write("Chúng ta thậm chí có thể tách ra nhiều channels hơn nữa, giống ví dụ dưới đây. Tuy nhiên, một vấn đề với Depthwise convolution chính là mỗi layer của output chỉ là tổng hợp thông tin của duy nhất 1 channel của input thôi (phần màu trắng trong hình ảnh dưới). Điều này không đúng so với CNN-based bình thường khi mà mỗi output pixel là tổng hợp của những pixels lân cận.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/depthwise_problem.jpg", caption="Depthwise convolution problem")
        
        st.write("Do đó, sau mỗi depthwise, chúng ta cần filter dạng pointwise để thông tin giữa các channels có thể trao đổi được với nhau. Kết hợp Depthwise + Pointwise convolution lại như thế chúng ta sẽ có được Depthwise separable convolution.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/pointwise.jpg", caption="Pointwise convolution")

    st.title("CAFormer")
    st.write("Tương tự như ConvFormer, chỉ khác là tác giả muốn kết hợp cả local và global representation. Tuy nhiên, nếu áp dụng attention để học global representation ngay từ input đầu vào thì kích thước ảnh lớn, sẽ tiêu tốn tài nguyên nhiều. Vì thế, họ đã đưa input vào CNN để tổng hợp local information (2 stages đầu), sau đó reshape và đưa vào Attention để học global.")
    _, c2, _ = st.columns(3)
    with c2:
        st.image("images/caformer.jpg", caption="Overall CAFormer framework")
    _, c2, _ = st.columns([3, 1, 3])
    with c2:
        st.image("images/caformer_block.jpg", caption="CAFormer block")
else: 
    # Transformer
    st.markdown('<a id="transformer"></a>', unsafe_allow_html=True)
    st.title("Transformer in natural language processing")
    st.write("In practice, a Transformer model actually takes in a sentence and predicts its next token (A token isn’t exactly a word, but for simplicity we treat 1 token ≈ 1 word)")
    st.write(
        "At first, because computers only understand numbers, not text, so we need to embed words into vectors. "
        "The meaning behind embedding word/ character to numbers is that each vocabulary term is essentially our rule to map a character to a numeric position plus semantic meaning. "
        "For instance, if we mapped 'h'→1, 'e'→2, 'l'→3, 'o'→4, then 'hello' would become '12334'. "
        "But because vocabularies are huge and semantics complex, we cannot use simple numbers to represent the word. So that's why high-dimensional embeddings were invented. Therefore, 'hello' can be '1234...863"
    )

    st.write(
        "After training a standalone embedding model, we feed in tokens and get their embeddings. "
        "Then we need positional embeddings. Without them, 'I love you' and 'You love I' map to the same vectors. "
        "Moreover, standalone embeddings lack sentence‐level context. "
        "For example, “The football match was exciting.” (match: a sport event) vs “He lit the fire with a match.” "
        "(match: a small stick for lighting a fire). Both are nouns but have very different meanings — yet a raw shared embedding model would treat them the same, which is incorrect."
    )

    st.write(
        "Therefore, the Transformer’s job is to adjust those embeddings according to context, yielding richer, context‐aware embeddings → the Attention mechanism."
    )
    st.info("Positional embeddings encode each token’s position in the sentence; Attention encodes relationships between tokens.")

    # Attention Mechanism
    st.markdown('<a id="attention-mechanism"></a>', unsafe_allow_html=True)
    st.title("Attention Mechanism")
    st.write(
        "To bring information from other tokens’ embeddings into a target token, we must decide how much each should contribute. "
        "Some tokens are crucial for context; others are irrelevant."
    )
    st.write(
        "We compute these weights via Query, Key, and Value. For each token, we examine other tokens to enrich its meaning. "
        "For humans, if the token is a noun, we might look for adjectives. Computers can’t do that intuitively, so the Query module asks: "
        "'What information do I need to request from other tokens?' That is the role of the Query."
    )
    st.write(
        "Similarly, the Key is the Query’s answer—it’s a brief description of each token (what it has), without revealing full content. "
        "We then take the dot product of Query and Key and apply softmax to get attention scores: each token’s contribution to the target."
    )
    st.write(
        "And we multiply the attention score with the content of corresponding token's embeddings. However, the actual content we want isn’t the token’s original embedding but its Value (a projection that compactly represents the token)."
    )
    st.write(
        "When we compute Query·Key, the target token is naturally considered, so its self-attention score is usually higher. "
        "We also apply masking. Since we predict the next token, we cannot see future tokens, so we only attend to past tokens."
    )
    st.image("images/querry-key.jpg")
    st.info(
        "All of the above is self-attention. Cross-attention uses Query from one dataset and Key from another "
        "(e.g., English↔Vietnamese text or audio waveform↔transcript). Everything else works the same."
    )
    st.info(
        "Multi-head attention runs multiple Q/K/V sets in parallel through single‐head attention modules, "
        "concatenates their outputs, and then optionally projects back to the desired dimension via a linear layer."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image('images/attention.jpg', caption="Self-Attention vs MLP")
    with col2:
        st.image('images/mlp.jpg',      caption="MLP Layer")

    st.write(
        "Here we can see the clear difference between Attention and a traditional MLP. "
        "In an Attention layer, all input embeddings exchange information to form output embeddings—"
        "each output token depends on many input tokens. "
        "In an MLP, each output token depends only on its own input embedding."
    )
    
    # MetaFormer
    st.markdown('<a id="metaformer"></a>', unsafe_allow_html=True)
    st.title("MetaFormer for Visual Tasks")

    st.write("**MetaFormer** is a general architecture for both computer vision and natural language processing, with two separate components: **Token Mixer** (handling interactions between spatial positions) and **Channel MLP** (handling interactions between channels).")

    # List variants
    st.write("""
    **Four main variants**:
    1. **IdentityFormer** – no mixer, baseline.  
    2. **RandFormer** – random mixer.  
    3. **ConvFormer** – uses convolution as a local mixer.  
    4. **CAFormer** – uses attention as a global mixer.  
    Among these, IdentityFormer and RandFormer are simple models used mainly to demonstrate the viability of the architecture.
    """)

    # Images
    _, c2, _ = st.columns([3,1,3])
    with c2:
        st.image("images/metaformer.jpg", caption='General structure of MetaFormer')
    _, c2, _ = st.columns([1,3,1])
    with c2:
        st.image("images/metaformer_benchmark.jpg", caption='MetaFormer benchmark')

    # Pseudo-code block
    st.markdown("**Formula of a MetaFormer block**:")
    st.code("""
    # I: input
    X     = InputEmbedding(I)
    X'    = X + TokenMixer( Norm1(X) )  # residual + token mixing
    X''   = X' + MLP_ch( Norm2(X') )    # residual + channel MLP
    """, language="python")

    # Compare CNN and Transformer
    st.write("""
    - **CNN**: a **local mixer**, mixing information within neighboring pixels.  
    - **Transformer**: a **global mixer**, aggregating context from all tokens using attention.  
    """)

    # Shape issue
    st.info(
        "MetaFormer handles the mismatch in shape between different stages and different mixers simply by **reshaping** data between stages: "
        "from H x W x C (CNN format) → (H·W) x C (Attention format) or vice versa, depending on the Token Mixer."
    )
    st.write("MetaFormer divides processing into four stages. The input is an image with shape H x W x 3, and the outputs at each following stage are:")
    _, c2, _ = st.columns(3)    
    with c2:
        st.image("images/stages.jpg")

    st.title("ConvFormer")
    st.write("In this model, the authors use an entirely CNN-based block design, where each block follows the MetaFormer structure described above, and the token mixer they use is convolution.")
    _, c2, _ = st.columns(3)
    with c2:
        st.image("images/convformer.jpg", caption="Overall ConvFormer framework")
    _, c2, _ = st.columns([3, 1, 3])
    with c2:
        st.image("images/convformer_block.jpg", caption="ConvFormer block")

    with st.expander("Depthwise separable convolution"):
        st.write("Here's how a normal CNN works. We have multiple filters (8 in the example below), and the input passes through each filter to produce outputs. The issue is that if the input size is large, the filter size also grows (current filter size: 8x3x3), leading to high resource consumption and slow computation.")
        _, c2, _ = st.columns([1, 3, 1])
        with c2:
            st.image("images/normal_cnn.jpg", caption="Normal CNN")
        
        st.write("Therefore, instead of using large filter sizes (8x3x3) with each filter spanning across all input channels, we split them into smaller filters (4x3x3) that only operate on a single channel. Although the number of filters stays the same, the size and number of parameters are significantly reduced.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/depthwise.jpg", caption="Depthwise convolution")
        
        st.write("We can even split into more channels, as shown in the example below. However, a problem with depthwise convolution is that each output layer only aggregates information from one input channel (illustrated by the white area in the image below). This is unlike a standard CNN where each output pixel aggregates information from neighboring pixels across channels.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/depthwise_problem.jpg", caption="Depthwise convolution problem")
        
        st.write("Thus, after each depthwise convolution, we need a pointwise convolution to allow information exchange between channels. Combining depthwise and pointwise convolution gives us what is called Depthwise Separable Convolution.")
        _, c2, _ = st.columns([1, 3, 1])    
        with c2:
            st.image("images/pointwise.jpg", caption="Pointwise convolution")

    st.title("CAFormer")
    st.write("Similar to ConvFormer, but here the authors aim to combine both local and global representations. However, if attention is used to capture global information right from the input stage, the large image size would lead to high resource consumption. Therefore, they first pass the input through CNNs to gather local information (in the first 2 stages), then reshape and apply Attention to learn global representations.")
    _, c2, _ = st.columns(3)
    with c2:
        st.image("images/caformer.jpg", caption="Overall CAFormer framework")
    _, c2, _ = st.columns([3, 1, 3])
    with c2:
        st.image("images/caformer_block.jpg", caption="CAFormer block")
