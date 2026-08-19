"""


from collections import Counter
from datetime import datetime
from pathlib import Path
import numbers
import re
import warnings

import jieba
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser
from gensim import corpora, models
from gensim.models import CoherenceModel, Word2Vec
from sklearn.cluster import KMeans
from tqdm import tqdm

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False



INPUT_FOLDER_NAME = "描述统计文档"
OUTPUT_FOLDER_NAME = "主题效应占比结果"
STOPWORDS_FILE_NAME = "stopwords.txt"
ADDWORDS_FILE_NAME = "addwords.txt"

DATE_COLUMN_CANDIDATES = (
    "date",
    "日期",
    "时间",
    "发布时间",
    "发布时刻",
    "time",
    "time_bin",
    "datetime",
    "timestamp",
)
TEXT_COLUMN_CANDIDATES = (
    "zhengwen",
    "正文",
    "文本",
    "内容",
    "text",
)

TIME_FREQUENCY = "30min"
METRIC_COLUMN = "topic_avg_centrality"
MAX_TOPICS = 15
ATTRIBUTE_CLUSTER_COUNT = 50
RANDOM_STATE = 42



def remove_timezone(timestamp):
    if pd.isna(timestamp):
        return pd.NaT
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def parse_numeric_date(value):

    numeric_value = float(value)

    if 1 <= numeric_value <= 100000:
        timestamp = pd.to_datetime(
            numeric_value,
            unit="D",
            origin="1899-12-30",
        )
        return remove_timezone(timestamp).round("s")

    absolute_value = abs(numeric_value)
    try:
        if 1e9 <= absolute_value < 1e11:
            return remove_timezone(
                pd.to_datetime(numeric_value, unit="s")
            )
        if 1e11 <= absolute_value < 1e14:
            return remove_timezone(
                pd.to_datetime(numeric_value, unit="ms")
            )
        if 1e14 <= absolute_value < 1e17:
            return remove_timezone(
                pd.to_datetime(numeric_value, unit="us")
            )
        if 1e17 <= absolute_value < 1e20:
            return remove_timezone(
                pd.to_datetime(numeric_value, unit="ns")
            )
    except Exception:
        return pd.NaT

    return pd.NaT


def normalize_chinese_datetime_text(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""

    text = (
        text.replace("／", "/")
        .replace("－", "-")
        .replace("：", ":")
        .replace("　", " ")
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d{4})\s*年\s*", r"\1-", text)
    text = re.sub(r"(\d{1,2})\s*月\s*", r"\1-", text)
    text = re.sub(r"(\d{1,2})\s*日", r"\1 ", text)
    text = re.sub(r"(\d{1,2})\s*时\s*", r"\1:", text)
    text = re.sub(r"(\d{1,2})\s*分\s*", r"\1:", text)
    text = re.sub(r"(\d{1,2}(?:\.\d+)?)\s*秒", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" :-")


def parse_chinese_date(date_value):
  
    if date_value is None:
        return pd.NaT

    if isinstance(
        date_value,
        (pd.Timestamp, datetime, np.datetime64),
    ):
        try:
            return remove_timezone(date_value)
        except Exception:
            return pd.NaT

    if isinstance(date_value, numbers.Number) and not isinstance(
        date_value,
        bool,
    ):
        return parse_numeric_date(date_value)

    raw_text = str(date_value).strip()
    if re.fullmatch(r"\d{8}", raw_text):
        return remove_timezone(
            pd.to_datetime(
                raw_text,
                format="%Y%m%d",
                errors="coerce",
            )
        )
    if re.fullmatch(r"\d{14}", raw_text):
        return remove_timezone(
            pd.to_datetime(
                raw_text,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )
        )
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw_text):
        return parse_numeric_date(float(raw_text))

    normalized_text = normalize_chinese_datetime_text(raw_text)
    if not normalized_text:
        return pd.NaT

    try:
        return remove_timezone(parser.parse(normalized_text))
    except Exception:
        return remove_timezone(
            pd.to_datetime(
                normalized_text,
                errors="coerce",
            )
        )



def clean_column_names(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in dataframe.columns
    ]
    return dataframe


def find_column(dataframe, candidates):
    column_map = {
        str(column).strip().casefold(): column
        for column in dataframe.columns
    }
    for candidate in candidates:
        matched_column = column_map.get(
            str(candidate).strip().casefold()
        )
        if matched_column is not None:
            return matched_column
    return None


def read_current_file(current_file):

    with pd.ExcelFile(current_file) as excel_file:
        checked_sheets = []

        for sheet_name in excel_file.sheet_names:
            dataframe = clean_column_names(
                pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                )
            )
            checked_sheets.append(
                f"{sheet_name}：{list(dataframe.columns)}"
            )

            date_column = find_column(
                dataframe,
                DATE_COLUMN_CANDIDATES,
            )
            text_column = find_column(
                dataframe,
                TEXT_COLUMN_CANDIDATES,
            )

            if date_column is None or text_column is None:
                continue

            dataframe = dataframe.rename(
                columns={
                    date_column: "date",
                    text_column: "zhengwen",
                }
            )
            return dataframe, {
                "sheet_name": sheet_name,
                "original_date_column": date_column,
                "original_text_column": text_column,
            }

    raise ValueError(
        f"{current_file.name}中没有同时包含date和zhengwen的工作表。"
        f"已检查：{'；'.join(checked_sheets)}"
    )


def find_numbered_workbooks(input_folder):

    numbered_files = []

    for file_path in Path(input_folder).iterdir():
        if not file_path.is_file():
            continue
        if file_path.name.startswith("~$"):
            continue

        match = re.fullmatch(
            r"(\d+)\.xlsx",
            file_path.name,
            flags=re.IGNORECASE,
        )
        if match:
            numbered_files.append(
                (int(match.group(1)), file_path)
            )

    numbered_files.sort(key=lambda item: item[0])
    return [file_path for _, file_path in numbered_files]


def load_stopwords(stopwords_path):
    with open(
        stopwords_path,
        "r",
        encoding="utf-8",
    ) as file:
        return {
            line.strip()
            for line in file
            if line.strip()
        }


def configure_jieba(addwords_path):

    if addwords_path.exists():
        jieba.load_userdict(str(addwords_path))
    jieba.initialize()


def clean_and_segment(text, stopwords):
    if not isinstance(text, str):
        return []

    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    words = jieba.lcut(text)

    return [
        word
        for word in words
        if (
            word not in stopwords
            and len(word) > 1
            and not re.search(r"[a-zA-Z0-9]", word)
        )
    ]


def find_optimal_topics(
    corpus,
    dictionary,
    texts,
    max_topics=MAX_TOPICS,
):
    nonempty_texts = [text for text in texts if text]
    if len(nonempty_texts) < 2:
        raise ValueError("有效文本不足2条，无法进行LDA主题数检验。")

    upper_topic_count = min(
        max_topics,
        len(dictionary),
        len(nonempty_texts),
    )
    if upper_topic_count < 2:
        raise ValueError("有效词项不足，无法训练至少2个主题。")

    best_score = -np.inf
    best_model = None
    best_topic_count = None
    evaluation_rows = []

    for topic_count in tqdm(
        range(2, upper_topic_count + 1),
        desc="LDA主题优化",
    ):
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=topic_count,
            passes=10,
            random_state=RANDOM_STATE,
            minimum_probability=0.0,
        )
        perplexity = float(
            np.exp2(lda_model.log_perplexity(corpus))
        )
        coherence = float(
            CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v",
            ).get_coherence()
        )
        score = coherence - 0.002 * perplexity

        evaluation_rows.append(
            {
                "主题数": topic_count,
                "困惑度": perplexity,
                "一致性": coherence,
                "综合得分": score,
            }
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_model = lda_model
            best_topic_count = topic_count

    if best_model is None:
        raise RuntimeError("LDA主题模型训练失败。")

    return (
        best_model,
        best_topic_count,
        pd.DataFrame(evaluation_rows),
    )


def get_topic_terms(lda_model, num_words=10):
    return {
        f"主题{topic_id + 1}": [
            word
            for word, _ in lda_model.show_topic(
                topic_id,
                num_words,
            )
        ]
        for topic_id in range(lda_model.num_topics)
    }


def assign_document_topics(lda_model, corpus):
    topic_numbers = []

    for document_bow in corpus:
        topic_distribution = lda_model.get_document_topics(
            document_bow,
            minimum_probability=0.0,
        )
        dominant_topic = max(
            topic_distribution,
            key=lambda item: item[1],
        )[0]
        topic_numbers.append(dominant_topic + 1)

    return topic_numbers


def build_attribute_clusters(texts):

    nonempty_texts = [text for text in texts if text]
    if not nonempty_texts:
        raise ValueError("当前文件没有可用于属性聚类的有效分词。")

    word2vec_model = Word2Vec(
        sentences=nonempty_texts,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        seed=RANDOM_STATE,
    )

    valid_words = sorted(
        {
            word
            for text in nonempty_texts
            for word in text
            if word in word2vec_model.wv
        }
    )
    if not valid_words:
        raise ValueError(
            "当前文件没有达到Word2Vec最小词频要求的属性词。"
        )

    actual_cluster_count = min(
        ATTRIBUTE_CLUSTER_COUNT,
        len(valid_words),
    )

    if actual_cluster_count == 1:
        word_clusters = {valid_words[0]: 0}
    else:
        word_vectors = np.asarray(
            [
                word2vec_model.wv[word]
                for word in valid_words
            ]
        )
        kmeans = KMeans(
            n_clusters=actual_cluster_count,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = kmeans.fit_predict(word_vectors)
        word_clusters = dict(zip(valid_words, labels))

    return word_clusters, actual_cluster_count


def calculate_network_metrics(topic_attribute_matrix):

    row_sums = topic_attribute_matrix.sum(axis=1)
    column_sums = topic_attribute_matrix.sum(axis=0)

    return {
        "density": float(
            np.mean(topic_attribute_matrix)
        ),
        "topic_avg_centrality": float(
            np.mean(row_sums)
        ),
        "topic_max_centrality": float(
            np.max(row_sums)
        ),
        "attribute_avg_centrality": float(
            np.mean(column_sums)
        ),
    }


def build_own_half_hour_metrics(
    dataframe,
    lda_model,
    dictionary,
    word_clusters,
):

    dataframe = dataframe.copy()
    dataframe["time_bin"] = dataframe["date"].dt.floor(
        TIME_FREQUENCY
    )

    first_time_bin = dataframe["time_bin"].min()
    last_time_bin = dataframe["time_bin"].max()
    time_bin_starts = pd.date_range(
        start=first_time_bin,
        end=last_time_bin,
        freq=TIME_FREQUENCY,
    )

    metric_rows = []
    topic_count = lda_model.num_topics

    for time_bin in tqdm(
        time_bin_starts,
        desc="当前文件半小时时间窗口",
    ):
        current_bin = dataframe[
            dataframe["time_bin"] == time_bin
        ]
        current_tokens = [
            word
            for document_tokens in current_bin["tokens"]
            for word in document_tokens
        ]

        topic_attribute_matrix = np.zeros(
            (
                topic_count,
                ATTRIBUTE_CLUSTER_COUNT,
            ),
            dtype=float,
        )

        if current_tokens:
            current_bow = dictionary.doc2bow(
                current_tokens
            )
            topic_distribution = (
                lda_model.get_document_topics(
                    current_bow,
                    minimum_probability=0.0,
                )
            )
            topic_probabilities = np.asarray(
                [
                    probability
                    for _, probability
                    in topic_distribution
                ],
                dtype=float,
            )
            word_counts = Counter(current_tokens)

            for word, word_count in word_counts.items():
                cluster_id = word_clusters.get(word)
                if cluster_id is None:
                    continue
                topic_attribute_matrix[
                    :,
                    cluster_id,
                ] += topic_probabilities * word_count

        metrics = calculate_network_metrics(
            topic_attribute_matrix
        )
        metrics.update(
            {
                "date": time_bin,
                "time_bin": time_bin,
                "文档数量": int(len(current_bin)),
            }
        )
        metric_rows.append(metrics)

    return pd.DataFrame(metric_rows)



def categorize_metric_effect(metric_dataframe):
    metric_dataframe = metric_dataframe.copy()
    first_quantile, second_quantile = metric_dataframe[
        METRIC_COLUMN
    ].quantile([0.33, 0.66])

    def categorize(value):
        if value <= first_quantile:
            return "低效应"
        if value <= second_quantile:
            return "中效应"
        return "高效应"

    metric_dataframe["effect"] = metric_dataframe[
        METRIC_COLUMN
    ].apply(categorize)
    return (
        metric_dataframe,
        float(first_quantile),
        float(second_quantile),
    )


def calculate_effect_proportion(
    dataframe,
    topic_col="topic",
    effect_col="effect",
):
    count_table = (
        dataframe.groupby(
            [topic_col, effect_col]
        )
        .size()
        .reset_index(name="count")
    )
    topic_total = (
        dataframe.groupby(topic_col)
        .size()
        .reset_index(name="topic_count")
    )
    result = pd.merge(
        count_table,
        topic_total,
        on=topic_col,
    )
    result["proportion"] = (
        result["count"]
        / result["topic_count"]
    )
    return result



def process_one_file(
    current_file,
    stopwords,
    output_root,
):
    file_number = current_file.stem
    file_output_folder = output_root / file_number
    file_output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataframe, source_info = read_current_file(
        current_file
    )
    original_row_count = len(dataframe)


    dataframe["date"] = dataframe["date"].apply(
        parse_chinese_date
    )
    invalid_date_count = int(
        dataframe["date"].isna().sum()
    )
    dataframe = dataframe.dropna(
        subset=["date"]
    ).copy()
    dataframe = dataframe.sort_values(
        "date"
    ).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError(
            f"{current_file.name}自身date列没有有效日期。"
        )


    dataframe["tokens"] = dataframe[
        "zhengwen"
    ].apply(
        lambda text: clean_and_segment(
            text,
            stopwords,
        )
    )
    texts = dataframe["tokens"].tolist()
    dictionary = corpora.Dictionary(texts)
    if len(dictionary) == 0:
        raise ValueError(
            f"{current_file.name}分词后没有有效词项。"
        )
    corpus = [
        dictionary.doc2bow(text)
        for text in texts
    ]

    (
        lda_model,
        topic_count,
        lda_evaluation,
    ) = find_optimal_topics(
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
    )
    dataframe["topic"] = assign_document_topics(
        lda_model,
        corpus,
    )
    topic_terms = get_topic_terms(
        lda_model,
        num_words=10,
    )


    (
        word_clusters,
        actual_cluster_count,
    ) = build_attribute_clusters(texts)


    metric_dataframe = build_own_half_hour_metrics(
        dataframe=dataframe,
        lda_model=lda_model,
        dictionary=dictionary,
        word_clusters=word_clusters,
    )
    (
        metric_dataframe,
        first_quantile,
        second_quantile,
    ) = categorize_metric_effect(
        metric_dataframe
    )


    dataframe["time_bin"] = dataframe[
        "date"
    ].dt.floor(TIME_FREQUENCY)
    analysis_dataframe = pd.merge(
        dataframe[
            [
                "date",
                "time_bin",
                "topic",
            ]
        ],
        metric_dataframe[
            [
                "time_bin",
                METRIC_COLUMN,
                "effect",
            ]
        ],
        on="time_bin",
        how="left",
        validate="many_to_one",
    )

    unmatched_count = int(
        analysis_dataframe["effect"].isna().sum()
    )
    if unmatched_count:
        raise RuntimeError(
            f"{current_file.name}内部仍有"
            f"{unmatched_count}条文本未匹配到自身时间窗。"
        )


    effect_detail = calculate_effect_proportion(
        analysis_dataframe,
        topic_col="topic",
        effect_col="effect",
    )
    effect_matrix = (
        effect_detail.pivot(
            index="topic",
            columns="effect",
            values="proportion",
        )
        .fillna(0)
        .reindex(
            columns=[
                "低效应",
                "中效应",
                "高效应",
            ],
            fill_value=0,
        )
    )

    topic_keywords = pd.DataFrame(
        {
            "主题编号": list(
                range(1, topic_count + 1)
            ),
            "关键词Top10": [
                ", ".join(
                    topic_terms[
                        f"主题{topic_number}"
                    ]
                )
                for topic_number
                in range(1, topic_count + 1)
            ],
        }
    )

    diagnostic = pd.DataFrame(
        [
            {
                "文件": current_file.name,
                "工作表": source_info["sheet_name"],
                "原日期列": source_info[
                    "original_date_column"
                ],
                "原文本列": source_info[
                    "original_text_column"
                ],
                "原始行数": original_row_count,
                "无效日期数": invalid_date_count,
                "有效行数": len(dataframe),
                "最早日期": dataframe["date"].min(),
                "最晚日期": dataframe["date"].max(),
                "半小时时间窗数": len(
                    metric_dataframe
                ),
                "成功匹配文本数": len(
                    analysis_dataframe
                ),
                "未匹配文本数": unmatched_count,
                "主题数": topic_count,
                "实际属性簇数": (
                    actual_cluster_count
                ),
                "低中效应分界值": first_quantile,
                "中高效应分界值": second_quantile,
            }
        ]
    )


    result_workbook = (
        file_output_folder
        / f"{file_number}_主题效应占比分析.xlsx"
    )
    with pd.ExcelWriter(result_workbook) as writer:
        diagnostic.to_excel(
            writer,
            sheet_name="运行诊断",
            index=False,
        )
        lda_evaluation.to_excel(
            writer,
            sheet_name="LDA主题评估",
            index=False,
        )
        topic_keywords.to_excel(
            writer,
            sheet_name="主题关键词",
            index=False,
        )
        metric_dataframe.to_excel(
            writer,
            sheet_name="自身半小时时序指标",
            index=False,
        )
        analysis_dataframe.to_excel(
            writer,
            sheet_name="文本主题与效应",
            index=False,
        )
        effect_detail.to_excel(
            writer,
            sheet_name="主题效应占比明细",
            index=False,
        )
        effect_matrix.to_excel(
            writer,
            sheet_name="主题效应占比矩阵",
        )


    plot_matrix = effect_matrix.copy()
    plot_matrix.index = [
        f"{topic_number}："
        + " ".join(
            topic_terms[
                f"主题{topic_number}"
            ][:5]
        )
        for topic_number in plot_matrix.index
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )
    sns.heatmap(
        plot_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        ax=axis,
    )
    axis.set_title(
        f"{current_file.name}主题×效应占比热力图"
    )
    axis.set_ylabel("主题（含前5个关键词）")
    axis.set_xlabel("效应等级")
    figure.tight_layout()

    heatmap_file = (
        file_output_folder
        / f"{file_number}_主题效应占比热力图.png"
    )
    figure.savefig(
        heatmap_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return {
        "file": current_file,
        "file_number": file_number,
        "date_start": dataframe["date"].min(),
        "date_end": dataframe["date"].max(),
        "valid_row_count": len(dataframe),
        "time_bin_count": len(metric_dataframe),
        "topic_count": topic_count,
        "attribute_cluster_count": (
            actual_cluster_count
        ),
        "topic_keywords": topic_keywords,
        "effect_matrix": effect_matrix,
        "result_workbook": result_workbook,
        "heatmap_file": heatmap_file,
    }


def print_one_file_result(
    result,
    current_index,
    total_count,
):

    print("\n" + "=" * 80)
    print(
        f"✅ [{current_index}/{total_count}] "
        f"{result['file'].name}计算完成"
    )
    print(
        "本次计算只使用："
        f"{result['file'].name}自身的date列"
    )
    print(
        f"自身日期范围：{result['date_start']}—"
        f"{result['date_end']}"
    )
    print(
        f"有效文本数：{result['valid_row_count']}；"
        f"自身半小时时间窗数："
        f"{result['time_bin_count']}"
    )
    print(
        f"最优主题数：{result['topic_count']}；"
        f"实际属性簇数："
        f"{result['attribute_cluster_count']}"
    )

    print(
        f"\n【{result['file'].name}主题关键词】"
    )
    print(
        result["topic_keywords"].to_string(
            index=False
        )
    )

    print(
        f"\n【{result['file'].name}主题×效应占比矩阵】"
    )
    print(
        result["effect_matrix"]
        .round(4)
        .to_string()
    )

    print(
        f"\n结果文件：{result['result_workbook']}"
    )
    print(
        f"热力图：{result['heatmap_file']}"
    )
    print("=" * 80 + "\n")



def main():
   
    script_folder = Path(__file__).resolve().parent
    input_folder = (
        script_folder / INPUT_FOLDER_NAME
    )
    output_root = (
        input_folder / OUTPUT_FOLDER_NAME
    )
    stopwords_path = (
        script_folder / STOPWORDS_FILE_NAME
    )
    addwords_path = (
        script_folder / ADDWORDS_FILE_NAME
    )

    if not input_folder.exists():
        raise FileNotFoundError(
            f"未找到文件夹：{input_folder}"
        )
    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"输入路径不是文件夹：{input_folder}"
        )
    if not stopwords_path.exists():
        raise FileNotFoundError(
            f"未找到停用词文件：{stopwords_path}"
        )

    current_files = find_numbered_workbooks(
        input_folder
    )
    if not current_files:
        raise FileNotFoundError(
            f"{input_folder}中没有找到"
            "1.xlsx、2.xlsx……"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    stopwords = load_stopwords(
        stopwords_path
    )
    configure_jieba(addwords_path)

    total_count = len(current_files)
    print("=" * 80)
    print(
        f"共发现{total_count}个编号文件："
    )
    print(
        " → ".join(
            file_path.name
            for file_path in current_files
        )
    )
    if total_count != 24:
        print(
            f"⚠️ 当前实际发现{total_count}个文件，"
            "程序仍会按实际编号顺序全部处理。"
        )
    print(
        "每个文件独立使用自己的date列；"
        "不读取任何外部指标文件。"
    )
    print("=" * 80)

    successful_results = []
    failed_files = []

    for current_index, current_file in enumerate(
        current_files,
        start=1,
    ):
        print(
            f"\n开始处理[{current_index}/{total_count}]："
            f"{current_file.name}"
        )

        try:
            result = process_one_file(
                current_file=current_file,
                stopwords=stopwords,
                output_root=output_root,
            )
        except Exception as error:
            failed_files.append(
                (
                    current_file.name,
                    str(error),
                )
            )
            print(
                f"❌ {current_file.name}处理失败："
                f"{error}"
            )
            print("继续处理下一个文件。")
            continue

        successful_results.append(result)

   
        print_one_file_result(
            result=result,
            current_index=current_index,
            total_count=total_count,
        )

    print("\n" + "=" * 80)
    print(
        f"批量处理结束：成功"
        f"{len(successful_results)}个，"
        f"失败{len(failed_files)}个。"
    )
    print(f"全部结果目录：{output_root}")

    if failed_files:
        print("\n失败文件：")
        for file_name, error_message in failed_files:
            print(
                f"- {file_name}："
                f"{error_message}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
