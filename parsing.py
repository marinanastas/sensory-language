from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, ProcessPoolExecutor
import json
import os
import re
import time
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import lxml
import nltk
import pandas as pd
from prereform2modern import Processor
import pymorphy2

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


morph = pymorphy2.MorphAnalyzer()
df_sensory_language = pd.read_csv("sensory_language_2.csv", index_col=[0])


def update_sensory_df():
    with open(f"complete_sensory_results_pre1950.json", "r", encoding="utf-8") as f:
        pre1950_results: dict[str, int] = json.load(f)
    df_sensory_language["pre1950"] = df_sensory_language["Слово"].map(pre1950_results)
    df_sensory_language.to_csv("sensory_language_pre1950.csv")


def count_lemmas(sentence: str) -> Counter:
    """
    Tokenize text, lemmatize it, then return word frequency counter.
    """
    return Counter([
        morph.parse(word)[0].normal_form
        for word in nltk.word_tokenize(sentence)
    ])


def replace_newlines(text: str) -> str:
    return re.sub(r"\n", " ", text)


# функция для обработки каждого XML-файла
def process_xml_file(xml_file_path: str) -> dict[str, int]:
    # открываем файл и парсим его
    print(f"parsing file {xml_file_path} size of {os.stat(xml_file_path).st_size} bytes")
    with open(xml_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    cleared_text = replace_newlines(soup.body.text)
    # modernized_text, changes, s_json = Processor.process_text(
    #     text=cleared_text,
    #     show=False,
    #     delimiters=False,
    #     check_brackets=False
    # )
    lemma_counts = count_lemmas(cleared_text)
    # print("LEMMA COUNTS ", len(lemma_counts))
    
    sensory_language_count = {
        row["Слово"]: lemma_counts[row["Слово"]]
        for index, row in df_sensory_language.iterrows()
        if row["Слово"] in lemma_counts
    }
    print(f"found {len(sensory_language_count)} words for file {xml_file_path}")
    return sensory_language_count


def main():
    complete_sensory_count = defaultdict(int)
    executor = ProcessPoolExecutor()
    futures: list[Future] = []

    # функция для обхода всех папок и подпапок
    def process_folder(folder_path: str):
        # получаем список файлов и папок в текущей директории
        file_list = os.listdir(folder_path)
        
        # обходим каждый файл/папку
        for file_name in file_list:
            # получаем полный путь к файлу/папке
            full_path = os.path.join(folder_path, file_name)
            
            # если это папка, то рекурсивно вызываем эту же функцию для неё
            if os.path.isdir(full_path):
                process_folder(full_path)
            # если это XML-файл, то вызываем функцию для его обработки
            elif os.path.isfile(full_path) and file_name.endswith(".xml"):
                futures.append(executor.submit(process_xml_file, full_path))

    dir_name = "post1950"
    start_time = time.time()
    process_folder(dir_name)
    for future in as_completed(futures):
        result: dict[str, int] = future.result()
        # print(f"COMPLETE RESULT {result}")
        for k, v in result.items():
            complete_sensory_count[k] += v

    with open(f"complete_sensory_results_post1950.json", "w", encoding="utf-8") as f:
        json.dump(complete_sensory_count, f, ensure_ascii=False, indent=4)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    hours_mod = total_time % 3600
    minutes = int(hours_mod // 60)
    seconds = int(hours_mod % 60)
    print(f"parsing took {hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    main()