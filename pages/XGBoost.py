import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="XGBoost")

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



####################################################################################################################################################
st.markdown('<h2 style="color:white;">XGBoost for Regression</h2>', unsafe_allow_html=True)

st.markdown('<h3 style="color:#98FB98;">Giới thiệu</h3>', unsafe_allow_html=True)

st.write("Chúng ta có dữ liệu về liều lượng thuốc dùng và mức độ hiệu quả của thuốc. Theo dataset thì khi sử dụng 30mg sẽ cho kết quả tốt nhất. Bài toán cần quan tâm là dự đoán liều lượng thuốc là bao nhiêu mg sẽ cho ra tác dụng tương ứng là bao nhiêu (nhập drug dossage(mg) -> drug effectiveness)")


# Dữ liệu
data = [[10, -10], [20, 5], [30, 7], [40, -5]]
predict = [[10, -2.65], [20, 2.15], [30, 2.15], [40, -1.15]]
x = [point[0] for point in data]
y = [point[1] for point in data]

pred_y = 0.5

fig = go.Figure()

# 1. Điểm thực tế (to hơn)
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=12, color='blue'),
    name='Sample'
))

# 2. Đường y = 0.5
fig.add_trace(go.Scatter(
    x=[min(x), max(x)],
    y=[pred_y, pred_y],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='y = 0.5'
))

# 3. Residuals và nhãn
for xi, yi in zip(x, y):
    # Vẽ đoạn residual
    fig.add_trace(go.Scatter(
        x=[xi, xi],
        y=[yi, pred_y],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        showlegend=False
    ))

    # Tính và hiện giá trị residual ở giữa đoạn
    residual = round(yi - pred_y, 2)
    mid_y = (yi + pred_y) / 2
    fig.add_trace(go.Scatter(
        x=[xi],
        y=[mid_y],
        mode='text',
        text=[f"{residual}"],
        textposition='middle right',
        showlegend=False
    ))

fig.update_layout(
    title='Dữ liệu mẫu',
    xaxis_title='Drug dossage',
    yaxis_title='Drug effectiveness',
    xaxis=dict(
        showgrid=False,
        color='black'  # Màu chữ trục X
    ),
    yaxis=dict(
        showgrid=False,
        color='black'  # Màu chữ trục Y
    ),
    font=dict(
        color='black',  # Màu chữ tiêu đề, residuals
        size=20
    ),
    # paper_bgcolor='white',
    plot_bgcolor='white'
)

# Hiển thị trong Streamlit
st.plotly_chart(fig)




st.markdown('<h3 style="color:#98FB98;">Giải thuật</h3>', unsafe_allow_html=True)

st.markdown('<h4 style="color:cyan;">Step 1:</h4>', unsafe_allow_html=True)
st.write("Chúng ta đặt dự đoán ban đầu (Initial prediction) là 0.5. Nghĩa là khi đưa bất kì drug dossage nào vào (vd 5mg, 50mg, 100mg) thì chúng ta đề dự đoán drug effectiveness là 0.5. và giá trị chúng ta có thể thay đổi, mặc định là 0.5")

st.markdown('<h4 style="color:cyan;">Step 2:</h4>', unsafe_allow_html=True)
st.write("Kế tiếp, tính tất cả độ sai lệch (residuals) giữa giá trị thật của các điểm dữ liệu (giá trị y của mỗi sample) và giá trị dự đoán (Initial prediction = 0.5) cho tất cả samples trong dataset: có được Residual = [-10.5, 4.5, 6.5, -5.5]")
st.write("Như vậy, mỗi tree sẽ bắt đầu như một leaf. Nghĩa là ta gom toàn bộ dữ liệu vào cùng 1 lá rồi mới bắt đầu phân chia giống như Decision tree")
_, col2, _ = st.columns([1, 3, 1])
with col2:
    st.image("images/initial_leaf.jpg", caption="Root của cây")

st.markdown('<h4 style="color:cyan;">Step 3:</h4>', unsafe_allow_html=True)
st.write("Để tiến hành chia tập dữ liệu, chúng ta cần xác định các threshold và quá trình đặt câu hỏi. Các threshold ở đây là drug dossage trung bình của 2 điểm dữ liệu kề nhau liên tiếp. Với các drug dossage hiện tại là [10, 20, 30, 40] thì ta có drug dossage threshold là [15, 25, 35]")
# 4. Thresholds (ví dụ ngưỡng tại x=15, 25, 35)
thresholds = [15, 25, 35]
for t in thresholds:
    fig.add_trace(go.Scatter(
        x=[t, t],
        y=[min(y)-5, max(y)+5],
        mode='lines',
        line=dict(color='purple', dash='dot', width=4),
        name=f'Threshold {t}'
    ))
st.plotly_chart(fig)
st.write("Có 3 threshold nên chúng ta sẽ có 3 cách chia cây. Đồng nghĩa với việc có 3 trường hợp. Và cũng giống như Decision tree, chúng ta cần lựa chọn chia theo trường hợp nào trước bằng cách tính Gini index or Information gain")
st.image("images/three_cases.jpg")

st.write("Ta có thể hiểu việc phân loại này là gom nhóm những samples tương đồng với nhau vào cùng 1 leaf. Ta cần đại lượng đo độ đồng nhất, nếu giá trị đó cao thì các samples đang có cùng xu hướng. Vì thế cần tính Similarity score theo công thức của XGBoost: ")
_, col2 = st.columns([1, 25])
with col2:
    st.markdown(r"Similarity score = $\frac{(\text{Sum of residuals})^2}{\text{Number of residuals} + \lambda}$, với $\lambda$ là regularization term")
st.write("Và từ đó tính độ hiệu quả của phép chia ~ Các node sau có Similarity score tốt hơn node hiện tại:")
_, col2 = st.columns([1, 15])
with col2:
    st.markdown(r"$\text{Gain}$ = $\text{Similarity}_{left}$ + $\text{Similarity}_{right}$ - $\text{Similarity}_{root}$")
st.markdown(r"Giả sử cho $\lambda = 0$ thì ta có kết quả như sau")
st.image("images/s_gain.jpg")
st.write("Theo hình trên thì chia theo trường hợp đầu tiên sẽ là có lợi nhất, vì vậy chúng ta chọn cách này.")
with st.expander("Regularization term"):
    st.write("Theo như hình trên thì những leaf có 1 sample trong đó thì Similarity score của leaf rất cao. Vì đây là đại lượng đo độ đồng nhất, nên nếu leaf chỉ có 1 sample thì chắc chắn không có sự bất đồng nhất nên sẽ cao. Tuy vậy, cần giảm tính nhạy cảm của mô hình đối với những dữ liệu đơn lẻ. Nếu không thì mô hình sẽ cố gắng tách riêng mỗi leaf là 1 sample -> dẫn đến overfitting")
    st.write("Như trong ví dụ dưới đây, khi không có regularization, mô hình sẽ chọn chia theo trường hợp trái (220 > 212.5), còn khi regularization = 1 thì sẽ chọn bên phải (140 < 141.6). Vì Similarity score của root là giống nhau nên không cần tính.")
    st.image("images/regularization.jpg")
    st.markdown(r"Như trong ví dụ trên thì khi $\lambda=0$ Similarity score của leaf chỉ có sample 10 là $\frac{10^2}{ 1 } = 100$ còn khi $\lambda = 1$ thì là  $\frac{10^2}{ 1+1 } = 50$")
with st.expander("Level-wise với Leaf-wise"):
    st.write("Với XGBoost, thuật toán dùng level-wise. Tức là ở mỗi tầng, sẽ xét tất cả các leaf. Ở mỗi leaf, duyệt qua từng features, mỗi features thực hiện giải thuật giống bên trên (lựa các threshold để tính toán). Tổng số trường hợp sẽ bằng số tổng số threshold ở các features. Rồi chọn ra trường hợp nào cho Gain tốt nhất.")
    st.write("Sau khi tìm được leaf mà có cách chia để Gain lớn nhất thì sẽ chia theo leaf đó là ra được 2 leaf con tầng dưới (tầng sâu hơn). Nhưng khi tiếp tục lựa chọn leaf để xét thì chỉ chọn leaf ở CÙNG TẦNG hiện tại, khi nào xong hết mới xét đến những leaf con vừa được tách ra. Ưu điểm là cây sinh đồng đều hơn và khó overfit (tránh tách quá sâu theo 1 nhánh nào có outlier gain)")
    st.write("Còn LightGBM sẽ sử dụng leaf-wise, nghĩa là trong mỗi lần tìm kiếm, thì thuật toán sẽ duyệt toàn bộ leaf đang có trong mô hình để tìm kiếm leaf có cách chia cho Gain lớn nhất. Miễn là có leaf tốt thì sẽ khai phá sâu hơn, không cần cùng cấp như XGBoost")

st.markdown('<h4 style="color:cyan;">Step 4:</h4>', unsafe_allow_html=True)
st.write("Thực hiện tương tự cho nhánh phải thì ta sẽ có kết quả sau. Giả sử chúng ta giới hạn số lượng samples tối thiểu mà mỗi leaf phải có là 2 thì đây sẽ là cây kết quả")
st.image("images/tree.jpg")

st.markdown('<h4 style="color:cyan;">Step 5: Cắt nhánh (pruning)</h4>', unsafe_allow_html=True)
st.markdown(r"Nếu để cây quá chi tiết, quá sâu có thể dẫn đến tăng độ phức tạp tính toán và overfit vào training data. Vì thế cần đặt một ngưỡng $\gamma$ để cắt bớt các nhánh theo cách sau. Xét những nhánh từ dưới lên (như ví dụ trên là từ nhánh Dosage<35 rồi đến nhánh Dosage<15), ta tính giá trị $Gain - \gamma$. Nếu dương thì giữ lại, nếu âm thì bỏ nhánh đó đi và gộp các leaf của nhánh đó thành một leaf chung)")
st.markdown(r"Ví dụ, nếu bỏ nhánh Dosage<35 đi thì ta sẽ có leaf bên phải của Dosage<15 là [4.5, 6.5, -5.5]. Và nếu trường hợp node cha có $Gain-\gamma$ âm (rơi vào trường hợp cắt nhánh) nhưng nhánh con của node đó lại có $Gain-\gamma$ dương thì vẫn giữ lại nguyên nhánh đó. Nghĩa là việc cắt nhánh diễn ra từ dưới lên trên")
st.markdown(r"Trong một số trường hợp thì Gain có thể âm (khi set regularization cao) thì khi đó, mặc dù set $\gamma=0$ nhưng $Gain - \gamma$ vẫn luôn âm nên việc cắt nhánh vẫn diễn ra")

st.markdown('<h4 style="color:cyan;">Step 6: Tính output cho leaf</h4>', unsafe_allow_html=True)
st.markdown("Đối với những leaf chứa nhiều samples (nhiều residuals của samples rơi vào chung 1 leaf) thì cần tính kết quả đại diện cho leaf đó như sau:")
_, col2 = st.columns([1, 15])
with col2:
    st.markdown(r"Output = $\frac{\text{Sum of residuals}}{\text{Number of residuals} + \lambda}$")
st.markdown(r"Công thức khá giống khi tính Similarity score, chỉ bỏ đi việc bình phương ở tử số. Và khi mà $\lambda = 0$ thì chính là trung bình cộng")
st.image("images/output.jpg", caption="XGBoost tree đầu tiên")
with st.expander("Ví dụ minh họa"):
    st.write("Giả sử ta muốn dự đoán drug effectiveness cho một liều lượng thuốc (drug dossage), ví dụ là 17mg, sẽ có kết quả là 5.5")
    st.image("images/calc_output.jpg")

st.markdown('<h4 style="color:cyan;">Step 6: Tính prediction mới</h4>', unsafe_allow_html=True)
st.markdown("Ban đầu ở step 1, khi chưa có gì thì chúng ta đã dự đoán drug effectiveness là 0.5 (Initial prediction) cho tất cả các drug dossage khác nhau. Bây giờ, sau khi đã có XGBoost tree đầu tiên, ta tính lại drug effectiveness cho từng drug dossage theo công thức như sau")

# _, col2 = st.columns([1, 5])
# with col2:
st.markdown(r"$\text{Prediction}$ = $\text{Initial prediction}$ + $\eta \cdot $ $\text{Kết quả từ XGBoost tree}_{ 1 }$")
_, col2 = st.columns([1, 4])
with col2:
    st.markdown(r"với $\eta$ được gọi là learning rate, mặc định 0.3")

# Dữ liệu gốc và dự đoán
data = [[10, -10], [20, 5], [30, 7], [40, -5]]
predict = [[10, -2.65], [20, 2.15], [30, 2.15], [40, -1.15]]

x = [point[0] for point in data]
y = [point[1] for point in data]
y_pred = [point[1] for point in predict]

fig = go.Figure()

# 1. Vẽ điểm thực tế
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=12, color='blue'),
    name='Ground truth'
))

# 2. Vẽ điểm dự đoán (màu xanh lá)
fig.add_trace(go.Scatter(
    x=x,
    y=y_pred,
    mode='markers',
    marker=dict(size=12, color='green', symbol='x'),
    name='Predict'
))

# 3. Vẽ residuals và hiển thị giá trị
for xi, yi, yhat in zip(x, y, y_pred):
    residual = round(yi - yhat, 2)
    mid_y = (yi + yhat) / 2

    # Vẽ đoạn residual
    fig.add_trace(go.Scatter(
        x=[xi, xi],
        y=[yi, yhat],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        showlegend=False
    ))

    # Hiển thị giá trị residual
    fig.add_trace(go.Scatter(
        x=[xi],
        y=[mid_y],
        mode='text',
        text=[f"{residual}"],
        textposition='middle right',
        showlegend=False
    ))

# Cập nhật layout
fig.update_layout(
    title='Residuals mới',
    xaxis_title='Drug dosage',
    yaxis_title='Drug effectiveness',
    xaxis=dict(showgrid=False, color='black'),
    yaxis=dict(showgrid=False, color='black'),
    font=dict(color='black', size=20),
    plot_bgcolor='white'
)

# Hiển thị
st.plotly_chart(fig)

st.write("Đây là kết quả khi tính prediction mới theo công thức trên. Có thể thấy, sự sai lệch giữa giá trị dự đoán và giá trị thực tế (residual)\
    được rút ngắn lại. Nếu không có learning rate, thì mô hình sẽ sửa lỗi quá mạnh, dễ overfit dữ liệu hoặc sai lệch.")
st.markdown("Kế đó, ta dùng những residuals này và thực hiện lại từ step 1 để sinh ra cây tiếp theo rồi cứ thế tiếp tục cho đến khi mô hình không \
    còn cải thiện nữa (residuals mới không giảm đáng kể so với residuals của cây hiện tại) hoặc là số lượng cây sinh ra đạt tối đa\
    được quyết định bởi hyper parameter **n_estimators**")

st.write("Đây chính là kết quả của step 2 cho lần xây XGBoost tree thứ 2")
_, col2, _ = st.columns([1, 3, 1])
with col2:
    st.image("images/new_leaf.jpg", caption="Root của cây mới")

with st.expander("Tính chất boosting"):
    st.write("Như vậy, cây tiếp theo được xây dựng dựa trên cây trước đó. Bởi vì mỗi cây được xây dựng từ residuals, mà residuals của cây được quyết định bởi prediction của các\
    cây trước đó, nên mỗi cây sẽ sửa những error của cây phía trước. Cho nên đây chính là tính chất boosting của XGBoost")
    
st.success(r"$\text{Pred}$ = $\text{Initial pred}$ + $\eta \cdot $ $\text{Tree}_{ 1 }$ + $\eta \cdot $ $\text{Tree}_{ 2 }$ + ... + $\eta \cdot $ $\text{Tree}_{ n\_estimators }$")
