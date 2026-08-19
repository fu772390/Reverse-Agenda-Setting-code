import pandas as pd
import numpy as np
import jieba
import re
from gensim import corpora, models
from gensim.models import Word2Vec, CoherenceModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import timedelta
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import warnings
from dateutil import parser


plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False    


warnings.filterwarnings('ignore')

def parse_chinese_date(date_str):
    """
    解析中文格式的日期字符串
    支持格式: "2018年07月22日 17:26" 或 "2018-07-22 17:26"
    """
    try:
      
        normalized_str = re.sub(
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'\1-\2-\3',
            str(date_str)
        )
        return parser.parse(normalized_str)
    except:
        return pd.NaT


def configure_jieba(addwords_path):
    print("🔄 加载自定义词典...")
    jieba.load_userdict(addwords_path)
    jieba.initialize()


def clean_and_segment(text, stopwords):
    if not isinstance(text, str):
        return []
  
    text = re.sub(r'http[s]?://\S+',  '', text)
    text = re.sub(r'[^\w\u4e00-\u9fff]',  ' ', text)

   
    words = jieba.lcut(text)


    filtered_words = []
    for word in words:
        if (
                word not in stopwords and
                len(word) > 1 and
                not re.search(r'[a-zA-Z0-9]', word) 
        ):
            filtered_words.append(word)

    return filtered_words


def load_stopwords(stopwords_path):
    print("🔄 加载停用词表...")
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])


def find_optimal_topics(corpus, dictionary, texts, max_topics=20):
    best_score = -np.inf
    best_lda = None
    best_num_topics = 0
    metrics = [] 

    print("🔍 寻找最优主题数...")
    for num_topics in tqdm(range(2, max_topics + 1), desc="LDA主题优化"):
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            random_state=42
        )
       
        perplexity = np.exp2(lda.log_perplexity(corpus))

      
        coherencemodel = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherencemodel.get_coherence()

      
        score = coherence - 0.002 * perplexity

        metrics.append({
            '主题数': num_topics,
            '困惑度': perplexity,
            '一致性': coherence
        })

        if score > best_score:
            best_score = score
            best_lda = lda
            best_num_topics = num_topics

    print(f"✅ 找到最优主题数: {best_num_topics} (一致性: {best_score:.4f})")
    return best_lda, best_num_topics, metrics


def get_topic_terms(lda_model, num_words=15):
    topics = {}
    for i in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(i,  num_words)
        topics[f"主题{i+1}"] = [word for word, _ in topic_terms]
    return topics

from sklearn.metrics import silhouette_score  # 放在文件顶部 if not already

def find_best_k_by_silhouette(word_vectors, k_range=range(10, 61)):
    best_k = None
    best_score = -1

    print("🔍 使用 Silhouette Score 自动评估最佳属性簇数...")
    for k in tqdm(k_range, desc="🔢 聚类评估中"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(word_vectors)
        score = silhouette_score(word_vectors, labels)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"✅ 最佳属性簇数为 {best_k}（轮廓系数最高，score={best_score:.4f}）")
    return best_k


def cluster_attribute_words(word_vectors, words, n_clusters=50):
    print("🔧 聚类属性词...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(word_vectors)
    return {word: cluster_id for word, cluster_id in zip(words, kmeans.labels_)}



def process_workbooks(stopwords_path='stopwords.txt',  addwords_path='addwords.txt'):
    print("="*50)
    print(f"📅 今天是2026年")
    print(f"🚀 开始处理工作表...")
    print("="*50)

 
    configure_jieba(addwords_path)
    stopwords = load_stopwords(stopwords_path)


    dfs = []
    for i in tqdm(range(1, 10), desc="📥 加载工作表"):
        df = pd.read_excel(f'{i}.xlsx')
        df['source'] = i
        dfs.append(df)
    combined_df = pd.concat(dfs)
    print(f"✅ 合并完成! 总记录数: {len(combined_df)}")


    print("🧹 文本清洗与分词中...")
    tqdm.pandas(desc="🔠  文本预处理")
    combined_df['cleaned'] = combined_df['zhengwen'].progress_apply(
        lambda x: clean_and_segment(x, stopwords))


    texts = combined_df['cleaned'].tolist()
    print("📚 创建词典与语料库...")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]


    lda_model, num_topics, lda_metrics = find_optimal_topics(corpus, dictionary, texts)
    topic_terms = get_topic_terms(lda_model)



    print("🤖 训练Word2Vec模型...")
    word2vec_model = Word2Vec(
        sentences=texts,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4
    )


    print("🧩 构建属性词簇...")
    all_words = list(set([word for text in texts for word in text]))
    valid_words = [word for word in all_words if word in word2vec_model.wv]
    word_vectors = np.array([word2vec_model.wv[word] for word in valid_words])


    best_k = find_best_k_by_silhouette(word_vectors, k_range=range(10, 61))


    word_clusters = cluster_attribute_words(word_vectors, valid_words, n_clusters=best_k)


    attribute_cluster_df = pd.DataFrame({
        '属性词': list(word_clusters.keys()),
        '聚类编号': list(word_clusters.values())
    })
    attribute_cluster_df.to_excel("属性词簇.xlsx", index=False)
    print("📁 已保存属性词簇至 属性词簇.xlsx")


    for source_num in range(1, 10):
        print("="*50)
        print(f"📊 处理工作表 {source_num}/9")
        print("="*50)


        df = pd.read_excel(f'{source_num}.xlsx')


        print("📆 解析日期列...")
        df['date'] = df['date'].apply(parse_chinese_date)


        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️ 警告: 发现 {invalid_dates} 条无效日期记录")
            df = df.dropna(subset=['date'])

        df = df.sort_values('date')


        df['time_bin'] = df['date'].dt.floor('30min')
        time_bins = pd.date_range(
            start=df['date'].min().floor('30min'),
            end=df['date'].max().ceil('30min'),
            freq='30min'
        )


        matrix_results = []
        metric_results = []


        print(f"⏳ 处理时间窗口 (共 {len(time_bins)-1} 个)")
        for i in tqdm(range(len(time_bins) - 1), desc="🕒 时间窗口处理"):
            start_time = time_bins[i]
            end_time = time_bins[i + 1]
            bin_df = df[(df['date'] >= start_time) & (df['date'] < end_time)]


            topic_attribute_matrix = np.zeros((num_topics,  50))

            if not bin_df.empty:

                all_texts = ' '.join(bin_df['zhengwen'].astype(str).tolist())
                cleaned_text = clean_and_segment(all_texts, stopwords)
                bow = dictionary.doc2bow(cleaned_text)


                topic_dist = lda_model.get_document_topics(bow,  minimum_probability=0)
                topic_probs = np.array([prob  for _, prob in topic_dist])


                for word in set(cleaned_text):
                    if word in word_clusters:
                        cluster_id = word_clusters[word]
                        if cluster_id < 50:  
                            word_count = cleaned_text.count(word)
                            for topic_id in range(num_topics):
                                topic_attribute_matrix[topic_id, cluster_id] += (
                                    topic_probs[topic_id] * word_count)


                attribute_cooccurrence_matrix = np.dot(topic_attribute_matrix.T, topic_attribute_matrix)


                attention_score = attribute_cooccurrence_matrix.sum()


                matrix_results.append({
                    'time_bin': start_time,
                    'topic_attribute_matrix': topic_attribute_matrix.flatten().tolist(),
                    'attribute_cooccurrence_matrix': attribute_cooccurrence_matrix.flatten().tolist(),
                    'time_attention': attention_score
                })





        matrix_df = pd.DataFrame(matrix_results)
        metric_df = pd.DataFrame(metric_results)


        topic_df = pd.DataFrame({
            '主题ID': [f"主题{i+1}" for i in range(num_topics)],
            '主题词': [', '.join(words) for words in topic_terms.values()]
        })


        lda_metrics_df = pd.DataFrame(lda_metrics)


        output_file = f'工作表{source_num}-1.xlsx'
        with pd.ExcelWriter(output_file) as writer:

            matrix_df.to_excel(writer,  sheet_name='主题属性矩阵', index=False)




            topic_df.to_excel(writer,  sheet_name='主题关键词', index=False)


            lda_metrics_df.to_excel(writer, sheet_name='LDA评估指标', index=False, float_format="%.10f")

        print(f"💾 已保存结果到: {output_file}")



if __name__ == '__main__':
    process_workbooks()
