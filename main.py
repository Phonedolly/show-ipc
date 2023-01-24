import argparse
import concurrent.futures
import requests
from requests import Response, Session
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from pandas import DataFrame
import seaborn as sns

NORM_FREQUENCY: int = 3000


def import_config() -> dict:
    print('import config...')
    with open("start.json", "r") as file:
        data: dict = json.load(file)
    return data


def fetch_search_result(query: str) -> Response:
    raw_from_browser = requests.get(
        'https://browser.geekbench.com/search?q=' + query)

    return raw_from_browser


def create_session() -> Session:
    print('create geekbench session')
    session = requests.Session()
    login_page: Response = session.get(
        'https://browser.geekbench.com/session/new')
    csrf_element = BeautifulSoup(login_page.content, features='lxml').select_one(
        '#wrap > div > div > div > div.card-body > form > input[type=hidden]:nth-child(2)')
    assert csrf_element is not None
    csrf: str = str(csrf_element.get('value'))

    with open('config.json', 'r') as f:
        user_data = json.load(f)
    form: dict = {
        'utf8': '✓',
        'authenticity_token': csrf,
        'user[username]': user_data['user_name'],
        'user[password]': user_data['password'],
        'commit': 'Log in'
    }
    session.post('https://browser.geekbench.com/session/create', data=form)

    return session


def fetch_each_result(each_url: str, session: Session) -> dict:
    result: dict = session.get(each_url).json()

    return result


def get_arguments() -> tuple[bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-search-cache', '-m',
                        help='use saved model data', action='store_true', dest='use_model_search_data')
    parser.add_argument('--sample-cache', '-s', help='use saved sample data',
                        action='store_true', dest='use_sample_data')

    use_model_search_data = parser.parse_args().use_model_search_data
    use_sample_data = parser.parse_args().use_sample_data

    return use_model_search_data, use_sample_data


if __name__ == "__main__":
    config: dict = import_config()
    urls: list[str] = []
    use_model_search_data, use_sample_data = get_arguments()
    cpus_bs4: list[BeautifulSoup] = []
    ipc_weights: list[float] = []
    ipc_weights_as_dataframe: Series
    bars: list[str] = []
    geekbench_session: Session = create_session()
    norm_ipc_weight: float = 0

    if use_model_search_data is False:
        for each_cpu in config['queries']:
            urls.append(each_cpu['query'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            result_from_queries: list = list(
                tqdm(executor.map(fetch_search_result, urls), total=len(urls)))

        for each_result in result_from_queries:
            cpu_bs4: BeautifulSoup = BeautifulSoup(
                each_result.content, features='lxml')
            file_name = cpu_bs4.select_one(
                '#wrap > div > div > div > div:nth-child(2) > div > form > div > input')
            assert file_name is not None
            file_name = file_name.get('value')
            cpus_bs4.append(cpu_bs4)
            with open(str(file_name) + '_query.html', 'w', encoding='utf-8') as f:
                f.write(cpu_bs4.prettify())

    else:
        for each_cpu in config['queries']:
            with open(each_cpu['query']+'_query.html', 'r', encoding='utf-8') as f:
                cpus_bs4.append(BeautifulSoup(f.read(), features='lxml'))

    for i in range(0, len(cpus_bs4)):
        # each_cpu = cpu_bs4.select_one(
        #     '#wrap > div > div > div > div:nth-child(3) > div.col-12.col-lg-9 > div.row > div:nth-child(1)')

        cpu_name: str = config['queries'][i]['description']
        search_result_for_each_cpu: list = cpus_bs4[i].select(
            '#wrap > div > div > div > div:nth-child(3) > div.col-12.col-lg-9 > div.row > div')
        sample_urls: list[str] = []
        samples: list[dict] = []
        ipc_weight_sum: float = 0
        average_ipc_weight: float = 0

        bars.append(cpu_name)

        if use_sample_data is False:
            for j in tqdm(range(0, 25), initial=1, total=25, desc='fetch ' + cpu_name + ' samples'):
                each_sample_url_a_element = search_result_for_each_cpu[j].select_one(
                    'div > div > div > div.col-12.col-lg-4 > a')
                assert each_sample_url_a_element is not None
                each_sample_url: str = 'https://browser.geekbench.com/' + \
                    str(each_sample_url_a_element.get('href')) + '.gb5'
                # sample_urls.append(each_sample_url)
                samples.append(fetch_each_result(
                    each_sample_url, geekbench_session))

                with open(cpu_name + '.' + str(j) + '.json', 'w') as f:
                    json.dump(samples[j], f, indent=2)

            # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            #     each_result_as_json: list[dict] = list(
            #         tqdm(executor.map(fetch_each_result, sample_urls, geekbench_session)))
        else:
            for j in tqdm(range(0, 25), initial=1, total=25, desc='load ' + cpu_name + ' samples'):
                with open(cpu_name + '.' + str(j) + '.json', 'r') as f:
                    samples.append(json.load(f))

        # 25 iterations
        for sample in samples:
            frequencies = pd.Series(
                sample['processor_frequency']['frequencies'])
            average_high_frequency: int = sum(frequencies.nlargest(5)) // 5
            if i == 0:
                ipc_weight: float = (sample['sections'][0]['score'] /
                                     average_high_frequency) * 4000
            else:
                ipc_weight: float = (sample['sections'][0]['score'] /
                                     average_high_frequency) * 4000
            ipc_weight_sum = ipc_weight_sum + ipc_weight

        average_ipc_weight = ipc_weight_sum / 25

        if i == 0:
            norm_ipc_weight = average_ipc_weight
        print('cpu name: ', cpu_name, ' average_ipc_weight', average_ipc_weight)
        ipc_weights.append( (average_ipc_weight / norm_ipc_weight) * 100)

    # plt.style.use('classic')
    # fig, ax = plt.subplots()

    # x_pos = np.arange(len(bars))
    # p = ax.bar(bars, ipc_weights, label='IPC')

    # ax.set_ylabel('IPC')
    # ax.set_title('IPC by CPU Generation')
    # ax.set_xticks(x_pos)
    # ax.legend()

    # ax.bar_label(p, label_type='center')
    # plt.show()
    dots = pd.Series(ipc_weights)

    sns.set_theme(context='paper', style='darkgrid',)
    sns.barplot(
        data=dots,
    
    )
    plt.show()
