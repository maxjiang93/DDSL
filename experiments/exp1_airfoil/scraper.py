import re
import os
import wget
from urllib.error import HTTPError as HTTPError
from tqdm import tqdm
import requests
import pandas as pd
from bs4 import BeautifulSoup

data_dir = "../data"

def my_wget(url, out):
    try:
        wget.download(url, out)
    except HTTPError:
        tqdm.write("Failed to download {0}".format(url))

def download_dat_and_polar():
    filename = 'airfoil_index.htm'
    textfile = open(filename, 'r')
    matches = []
    reg = re.compile("(?<=/airfoil/details\?airfoil=).*?(?=\")")
    for line in textfile:
        matches += reg.findall(line)
    textfile.close()
    matches = set(matches)  # make it unique, no duplicate entries
    matches = list(matches)

    names_ = matches
    names = []
    for match in matches:
        names.append(re.search(".*(?=-)", match).group(0))

    # create data folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pbar = tqdm(total=len(names))

    # go through models
    for model, model_ in zip(names, names_):
        # create sub folder for this airfoil
        sub_dir = os.path.join(data_dir, model)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        url_dat = "http://airfoiltools.com/airfoil/seligdatfile?airfoil=" + model_
        my_wget(url_dat, out=sub_dir)
        Re = [50000, 100000, 200000, 500000, 1000000]
        prefix = "http://airfoiltools.com/polar/csv?polar=xf-"
        suffix = "-n5"
        for rey in Re:
            url_n5 = prefix + model_ + "-" + str(rey) + suffix
            url_n9 = prefix + model_ + "-" + str(rey)
            my_wget(url_n5, out=sub_dir)
            my_wget(url_n9, out=sub_dir)
        pbar.update()

def scrape_info():
    filename = 'airfoil_index.htm'
    textfile = open(filename, 'r')
    matches = []
    reg = re.compile("(?<=/airfoil/details\?airfoil=).*?(?=\")")
    for line in textfile:
        matches += reg.findall(line)
    textfile.close()
    matches = set(matches)  # make it unique, no duplicate entries
    matches = list(matches)

    names_ = matches
    names = []
    for match in matches:
        names.append(re.search(".*(?=-)", match).group(0))

    # create data folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    pbar = tqdm(total=len(names))

    df = []

    # go through models
    for model, model_ in zip(names, names_):
        # create sub folder for this airfoil
        sub_dir = os.path.join(data_dir, model)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        url_dat = "http://airfoiltools.com/airfoil/details?airfoil=" + model_
        res = requests.get(url_dat)
        reg_max_thick_val = re.compile("(?<=Max thickness )(\d+?|\d+?\.\d+)(?=\% at)")
        reg_max_thick_pos = re.compile("(?<=at )(\d+?|\d+?\.\d+)(?=\% chord\.<br />)")
        reg_max_camber_val = re.compile("(?<=Max camber )(\d+?|\d+?\.\d+)(?=\% at)")
        reg_max_camber_pos = re.compile("(?<=at )(\d+?|\d+?\.\d+)(?=% chord<br />)")
        max_thick_val = float(reg_max_thick_val.findall(res.text)[0])
        max_thick_pos = float(reg_max_thick_pos.findall(res.text)[0])
        max_camber_val = float(reg_max_camber_val.findall(res.text)[0])
        max_camber_pos = float(reg_max_camber_pos.findall(res.text)[0])
        df.append([model, max_thick_val, max_thick_pos, max_camber_val, max_camber_pos])
        pbar.update()

    df = pd.DataFrame(df, columns=["id", "max_thick_val", "max_thick_pos", "max_camber_val", "max_camber_pos"])
    df.to_csv(os.path.join(data_dir, "properties.csv"), index=False)

if __name__ == "__main__":
    scrape_info()

