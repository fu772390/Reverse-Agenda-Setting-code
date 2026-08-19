import pandas as pd
import numpy as np
import jieba
import re
from gensim import corpora, models
from gensim.models import CoherenceModel
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
 
def parse_chinese_date(date_str):
    try:
        normalized_str = re.sub(r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3', str(date_str))
        return parser.parse(normalized_str)
    except:
        return pd.NaT
 
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])
 
def clean_and_segment(text, stopwords):
    if not isinstance(text, str):
        return []
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    words = jieba.lcut(text)
    return [w for w in words if w not in stopwords and len(w) > 1 and not re.search(r'[a-zA-Z0-9]', w)]

 
def find_optimal_topics(corpus, dictionary, texts, max_topics=15):
    best_score = -np.inf
    best_lda = None
    best_num_topics = 0
    for num_topics in tqdm(range(2, max_topics + 1), desc="LDA主题优化"):
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
        perplexity = np.exp2(lda.log_perplexity(corpus))
        coherencemodel = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherencemodel.get_coherence()
        score = coherence - 0.002 * perplexity
        if score > best_score:
            best_score = score
            best_lda = lda
            best_num_topics = num_topics
    print(f"✅ 最优主题数: {best_num_topics}")
    return best_lda, best_num_topics

 
def get_topic_terms(lda_model, num_words=10):
    topics = {}
    for i in range(lda_model.num_topics):
        topic_terms = lda_model.show_topic(i, num_words)
        topics[f"主题{i+1}"] = [word for word, _ in topic_terms]
    return topics

 
def calculate_effect_proportion(df, topic_col='topic', effect_col='effect'):
    count_table = df.groupby([topic_col, effect_col]).size().reset_index(name='count')
    topic_total = df.groupby(topic_col).size().reset_index(name='topic_count')
    merged = pd.merge(count_table, topic_total, on=topic_col)
    merged['proportion'] = merged['count'] / merged['topic_count']
    return merged

 
def main():
 
    stopwords_path = "stopwords.txt"
    text_file = "12.xlsx"
    metric_file = "1-2.xlsx"
    metric_col = "topic_avg_centrality"   

 
    stopwords = load_stopwords(stopwords_path)
    df_text = pd.read_excel(text_file)
    df_text['date'] = df_text['date'].apply(parse_chinese_date)
    df_text['tokens'] = df_text['zhengwen'].apply(lambda x: clean_and_segment(x, stopwords))

 
    dictionary = corpora.Dictionary(df_text['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df_text['tokens']]
    lda_model, num_topics = find_optimal_topics(corpus, dictionary, df_text['tokens'])
    topic_matrix = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    df_text['topic'] = [t+1 for t in topic_matrix]  

 
    topic_terms = get_topic_terms(lda_model, num_words=10)
    topic_keywords = pd.DataFrame({
        '主题编号': [i+1 for i in range(len(topic_terms))],
        '关键词Top10': [", ".join(words) for words in topic_terms.values()]
    })
    topic_keywords.to_excel("主题_关键词Top10.xlsx", index=False)

 
    df_metric = pd.read_excel(metric_file)
    df_metric['date'] = df_metric['date'].apply(parse_chinese_date)
    q1, q2 = df_metric[metric_col].quantile([0.33, 0.66])
    def categorize_effect(x):
        if x <= q1: return '低效应'
        elif x <= q2: return '中效应'
        else: return '高效应'
    df_metric['effect'] = df_metric[metric_col].apply(categorize_effect)

 
    df = pd.merge(df_text[['date','topic']], df_metric[['date','effect']], on='date', how='inner')

 
    result = calculate_effect_proportion(df, 'topic', 'effect')
    result.to_excel("主题_效应占比明细.xlsx", index=False)
    pivot = result.pivot(index='topic', columns='effect', values='proportion').fillna(0)
    pivot.to_excel("主题_效应占比矩阵.xlsx")

 
    topic_labels = [f"{idx}：" + " ".join(topic_terms[f'主题{idx}'][:5]) for idx in pivot.index]
    pivot.index = topic_labels

 
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title("主题×效应占比热力图（含主题关键词）")
    plt.ylabel("主题（含Top关键词）")
    plt.xlabel("效应等级")
    plt.tight_layout()
    plt.savefig("主题_效应占比热力图.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()