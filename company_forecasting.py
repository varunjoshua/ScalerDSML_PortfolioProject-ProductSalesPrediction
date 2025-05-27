{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOSMD/priQIaRWXV5A+ud2R",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/varunjoshua/ScalerDSML-ProductSalesForecast/blob/main/company_forecasting.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "djHOWOZwx5hz"
      },
      "outputs": [],
      "source": [
        "!git config --global user.email \"varun.joshua@gmail.com\"\n",
        "!git config --global user.name \"varunjoshua\""
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/varunjoshua/ScalerDSML-ProductSalesForecast.git\n",
        "%cd ScalerDSML-ProductSalesForecast"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1uPbvV86yIi6",
        "outputId": "f5fdbcf0-f056-4592-869b-b03aca4724bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'ScalerDSML-ProductSalesForecast'...\n",
            "remote: Enumerating objects: 29, done.\u001b[K\n",
            "remote: Counting objects: 100% (29/29), done.\u001b[K\n",
            "remote: Compressing objects: 100% (26/26), done.\u001b[K\n",
            "remote: Total 29 (delta 13), reused 0 (delta 0), pack-reused 0 (from 0)\u001b[K\n",
            "Receiving objects: 100% (29/29), 7.11 MiB | 10.32 MiB/s, done.\n",
            "Resolving deltas: 100% (13/13), done.\n",
            "/content/ScalerDSML-ProductSalesForecast\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Import Libararies/ Modules"
      ],
      "metadata": {
        "id": "hGPR70zr2KIE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install skforecast\n",
        "!pip install prophet"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "IQ8ebiDy31rR",
        "outputId": "ccdd88f9-7f27-4a42-9043-1707b7df6d9d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting skforecast\n",
            "  Downloading skforecast-0.16.0-py3-none-any.whl.metadata (16 kB)\n",
            "Requirement already satisfied: numpy>=1.24 in /usr/local/lib/python3.11/dist-packages (from skforecast) (2.0.2)\n",
            "Requirement already satisfied: pandas>=1.5 in /usr/local/lib/python3.11/dist-packages (from skforecast) (2.2.2)\n",
            "Requirement already satisfied: tqdm>=4.57 in /usr/local/lib/python3.11/dist-packages (from skforecast) (4.67.1)\n",
            "Requirement already satisfied: scikit-learn>=1.2 in /usr/local/lib/python3.11/dist-packages (from skforecast) (1.6.1)\n",
            "Collecting optuna>=2.10 (from skforecast)\n",
            "  Downloading optuna-4.3.0-py3-none-any.whl.metadata (17 kB)\n",
            "Requirement already satisfied: joblib>=1.1 in /usr/local/lib/python3.11/dist-packages (from skforecast) (1.5.0)\n",
            "Requirement already satisfied: numba>=0.59 in /usr/local/lib/python3.11/dist-packages (from skforecast) (0.60.0)\n",
            "Requirement already satisfied: rich>=13.9 in /usr/local/lib/python3.11/dist-packages (from skforecast) (13.9.4)\n",
            "Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /usr/local/lib/python3.11/dist-packages (from numba>=0.59->skforecast) (0.43.0)\n",
            "Collecting alembic>=1.5.0 (from optuna>=2.10->skforecast)\n",
            "  Downloading alembic-1.15.2-py3-none-any.whl.metadata (7.3 kB)\n",
            "Collecting colorlog (from optuna>=2.10->skforecast)\n",
            "  Downloading colorlog-6.9.0-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from optuna>=2.10->skforecast) (24.2)\n",
            "Requirement already satisfied: sqlalchemy>=1.4.2 in /usr/local/lib/python3.11/dist-packages (from optuna>=2.10->skforecast) (2.0.40)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from optuna>=2.10->skforecast) (6.0.2)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.5->skforecast) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.5->skforecast) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.5->skforecast) (2025.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich>=13.9->skforecast) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich>=13.9->skforecast) (2.19.1)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=1.2->skforecast) (1.15.3)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=1.2->skforecast) (3.6.0)\n",
            "Requirement already satisfied: Mako in /usr/lib/python3/dist-packages (from alembic>=1.5.0->optuna>=2.10->skforecast) (1.1.3)\n",
            "Requirement already satisfied: typing-extensions>=4.12 in /usr/local/lib/python3.11/dist-packages (from alembic>=1.5.0->optuna>=2.10->skforecast) (4.13.2)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich>=13.9->skforecast) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas>=1.5->skforecast) (1.17.0)\n",
            "Requirement already satisfied: greenlet>=1 in /usr/local/lib/python3.11/dist-packages (from sqlalchemy>=1.4.2->optuna>=2.10->skforecast) (3.2.2)\n",
            "Downloading skforecast-0.16.0-py3-none-any.whl (814 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m815.0/815.0 kB\u001b[0m \u001b[31m15.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading optuna-4.3.0-py3-none-any.whl (386 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m386.6/386.6 kB\u001b[0m \u001b[31m21.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading alembic-1.15.2-py3-none-any.whl (231 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m231.9/231.9 kB\u001b[0m \u001b[31m15.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading colorlog-6.9.0-py3-none-any.whl (11 kB)\n",
            "Installing collected packages: colorlog, alembic, optuna, skforecast\n",
            "Successfully installed alembic-1.15.2 colorlog-6.9.0 optuna-4.3.0 skforecast-0.16.0\n",
            "Requirement already satisfied: prophet in /usr/local/lib/python3.11/dist-packages (1.1.6)\n",
            "Requirement already satisfied: cmdstanpy>=1.0.4 in /usr/local/lib/python3.11/dist-packages (from prophet) (1.2.5)\n",
            "Requirement already satisfied: numpy>=1.15.4 in /usr/local/lib/python3.11/dist-packages (from prophet) (2.0.2)\n",
            "Requirement already satisfied: matplotlib>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from prophet) (3.10.0)\n",
            "Requirement already satisfied: pandas>=1.0.4 in /usr/local/lib/python3.11/dist-packages (from prophet) (2.2.2)\n",
            "Requirement already satisfied: holidays<1,>=0.25 in /usr/local/lib/python3.11/dist-packages (from prophet) (0.72)\n",
            "Requirement already satisfied: tqdm>=4.36.1 in /usr/local/lib/python3.11/dist-packages (from prophet) (4.67.1)\n",
            "Requirement already satisfied: importlib-resources in /usr/local/lib/python3.11/dist-packages (from prophet) (6.5.2)\n",
            "Requirement already satisfied: stanio<2.0.0,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from cmdstanpy>=1.0.4->prophet) (0.5.1)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.11/dist-packages (from holidays<1,>=0.25->prophet) (2.9.0.post0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (1.3.2)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (4.58.0)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (1.4.8)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (24.2)\n",
            "Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (11.2.1)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.0.0->prophet) (3.2.3)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.0.4->prophet) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.0.4->prophet) (2025.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil->holidays<1,>=0.25->prophet) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import (\n",
        "    mean_squared_error as mse,\n",
        "    mean_absolute_error as mae,\n",
        "    mean_absolute_percentage_error as mape)\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.linear_model import Ridge\n",
        "from xgboost import XGBRegressor\n",
        "from skforecast.recursive import ForecasterRecursive\n",
        "from skforecast.preprocessing import RollingFeatures\n",
        "from statsmodels.tsa.arima.model import ARIMA\n",
        "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
        "from prophet import Prophet\n",
        "from itertools import product\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "pd.set_option('display.float_format', '{:,.2f}'.format)\n",
        "import logging\n",
        "logging.getLogger('prophet').setLevel(logging.WARNING)\n",
        "logging.getLogger('cmdstanpy').disabled = True"
      ],
      "metadata": {
        "id": "iyu8Cv_eysFk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!gdown https://raw.githubusercontent.com/varunjoshua/ScalerDSML-ProductSalesForecast/main/data/ts_co.csv"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6w5nWhobAFTT",
        "outputId": "0701231b-51d0-4600-8225-f5b36d0da719"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading...\n",
            "From: https://raw.githubusercontent.com/varunjoshua/ScalerDSML-ProductSalesForecast/main/data/ts_co.csv\n",
            "To: /content/ScalerDSML-ProductSalesForecast/ts_co.csv\n",
            "\r  0% 0.00/7.22k [00:00<?, ?B/s]\r25.3kB [00:00, 37.7MB/s]       \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ts = pd.read_csv('ts_co.csv')\n",
        "ts.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "-Qhhwzj13KL2",
        "outputId": "af3572d3-f3f8-43bf-bf68-285b4c09774b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         Date  Holiday  Discounted Stores  Orders         Sales\n",
              "0  2018-01-01        1               1.00   19666 15,345,484.50\n",
              "1  2018-01-02        0               1.00   25326 19,592,415.00\n",
              "2  2018-01-03        0               1.00   24047 18,652,527.00\n",
              "3  2018-01-04        0               1.00   25584 19,956,267.00\n",
              "4  2018-01-05        0               1.00   28436 22,902,651.00"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8135e893-fa36-40ee-895b-2da693c274f2\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Date</th>\n",
              "      <th>Holiday</th>\n",
              "      <th>Discounted Stores</th>\n",
              "      <th>Orders</th>\n",
              "      <th>Sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2018-01-01</td>\n",
              "      <td>1</td>\n",
              "      <td>1.00</td>\n",
              "      <td>19666</td>\n",
              "      <td>15,345,484.50</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2018-01-02</td>\n",
              "      <td>0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>25326</td>\n",
              "      <td>19,592,415.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2018-01-03</td>\n",
              "      <td>0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>24047</td>\n",
              "      <td>18,652,527.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2018-01-04</td>\n",
              "      <td>0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>25584</td>\n",
              "      <td>19,956,267.00</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2018-01-05</td>\n",
              "      <td>0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>28436</td>\n",
              "      <td>22,902,651.00</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8135e893-fa36-40ee-895b-2da693c274f2')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8135e893-fa36-40ee-895b-2da693c274f2 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8135e893-fa36-40ee-895b-2da693c274f2');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-1a4d18e9-90b8-4df1-b19e-470ce04a52b0\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-1a4d18e9-90b8-4df1-b19e-470ce04a52b0')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-1a4d18e9-90b8-4df1-b19e-470ce04a52b0 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "ts",
              "summary": "{\n  \"name\": \"ts\",\n  \"rows\": 516,\n  \"fields\": [\n    {\n      \"column\": \"Date\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"num_unique_values\": 516,\n        \"samples\": [\n          \"2018-11-01\",\n          \"2019-05-16\",\n          \"2019-03-18\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Holiday\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Discounted Stores\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.3801871875195098,\n        \"min\": 0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 170,\n        \"samples\": [\n          0.1616438356164383,\n          0.6082191780821918\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Orders\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 4343,\n        \"min\": 2940,\n        \"max\": 39266,\n        \"num_unique_values\": 507,\n        \"samples\": [\n          28571,\n          23359\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Sales\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3270532.4449256393,\n        \"min\": 1762137.57,\n        \"max\": 26870817.0,\n        \"num_unique_values\": 516,\n        \"samples\": [\n          16137054.0,\n          21445713.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Pre-processing functions**"
      ],
      "metadata": {
        "id": "1FhpbjutBxhV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Function to process data for Regression**"
      ],
      "metadata": {
        "id": "QMrLYlnkCuUm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to tranform and process the inference data given\n",
        "  # The test data provided is ungrouped with records of all stores for each day\n",
        "  # The data needs to be grouped and transformed for the Recursive Forecasting function\n",
        "  # The function will group and aggregate the data for Company and Regions : R1, R2, R3, R4\n",
        "\n",
        "def inference_data_processor(data):\n",
        "    import pandas as pd\n",
        "    import numpy as np\n",
        "\n",
        "    # Step 1: Convert 'Date' to datetime and add 'Discounted_Flag'\n",
        "    data['Date'] = pd.to_datetime(data['Date'])\n",
        "    data['Discounted_Flag'] = data['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)\n",
        "\n",
        "    # Step 2: function to process each group\n",
        "    def process_group(group_df):\n",
        "        group_df = group_df.groupby('Date').agg({\n",
        "            'Holiday': 'last',\n",
        "            'Discounted_Flag': lambda x: x.sum() / x.count()\n",
        "        }).rename(columns={'Discounted_Flag': 'Discounted Stores'})\n",
        "\n",
        "        # Date features\n",
        "        group_df['Day Count'] = (group_df.index - group_df.index.min()).days\n",
        "        group_df['Weekend'] = group_df.index.dayofweek.isin([5, 6]).astype(int)\n",
        "        day_of_week = group_df.index.dayofweek\n",
        "        month = group_df.index.month\n",
        "\n",
        "        # Cyclical features\n",
        "        group_df['Month_sine'] = np.sin(2 * np.pi * month / 12)\n",
        "        group_df['Month_cosine'] = np.cos(2 * np.pi * month / 12)\n",
        "        group_df['Day of Week_sine'] = np.sin(2 * np.pi * day_of_week / 7)\n",
        "        group_df['Day of Week_cosine'] = np.cos(2 * np.pi * day_of_week / 7)\n",
        "\n",
        "        return group_df\n",
        "\n",
        "    # Step 3: Create datasets\n",
        "    inf_all = process_group(data)\n",
        "    inf_r1 = process_group(data[data['Region_Code'] == 'R1'])\n",
        "    inf_r2 = process_group(data[data['Region_Code'] == 'R2'])\n",
        "    inf_r3 = process_group(data[data['Region_Code'] == 'R3'])\n",
        "    inf_r4 = process_group(data[data['Region_Code'] == 'R4'])\n",
        "\n",
        "    return inf_all, inf_r1, inf_r2, inf_r3, inf_r4\n",
        "\n"
      ],
      "metadata": {
        "id": "EmRHZM4_YC-9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Function to process data for SARIMAX**"
      ],
      "metadata": {
        "id": "Rj5ifBa8C8Py"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Function inference_exog_processor will be used to transform and process the inference dataset...\n",
        "#...and return datframes with exog variable for Company and Regions : R1, R2, R3, R4, for the inference period\n",
        "\n",
        "def inference_exog_processor(data):\n",
        "\n",
        "    # Step 1: Convert 'Date' to datetime and add 'Discounted_Flag'\n",
        "    data['Date'] = pd.to_datetime(data['Date'])\n",
        "    data['Discounted_Flag'] = data['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)\n",
        "\n",
        "    # Step 2: function to process each group\n",
        "    def process_group(group_df):\n",
        "        group_df = group_df.groupby('Date').agg({\n",
        "            'Holiday': 'last',\n",
        "            'Discounted_Flag': lambda x: x.sum() / x.count()\n",
        "        }).rename(columns={'Discounted_Flag': 'Discounted Stores'})\n",
        "\n",
        "        return group_df\n",
        "\n",
        "    # Step 3: Creating datasets\n",
        "    exog_all = process_group(data)\n",
        "    exog_r1 = process_group(data[data['Region_Code'] == 'R1'])\n",
        "    exog_r2 = process_group(data[data['Region_Code'] == 'R2'])\n",
        "    exog_r3 = process_group(data[data['Region_Code'] == 'R3'])\n",
        "    exog_r4 = process_group(data[data['Region_Code'] == 'R4'])\n",
        "\n",
        "    return exog_all, exog_r1, exog_r2, exog_r3, exog_r4\n",
        "\n"
      ],
      "metadata": {
        "id": "hZl5FngsC7zN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Function to process data for Prophet**"
      ],
      "metadata": {
        "id": "7E5Z7-GMEE1L"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def prophet_data_formatter(data, is_inference=False):\n",
        "\n",
        "    df = pd.DataFrame()\n",
        "\n",
        "    if not is_inference:\n",
        "        df['y'] = data['Sales']\n",
        "\n",
        "    df['ds'] = pd.to_datetime(data['Date'])\n",
        "    exog = data.drop(['Date'] + ([] if is_inference else ['Sales']), axis=1)\n",
        "    df = pd.concat([df, exog.reset_index(drop=True)], axis=1)\n",
        "    return df"
      ],
      "metadata": {
        "id": "6jvcBPvgEEWH"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Model Params**"
      ],
      "metadata": {
        "id": "sA0Kwt87VUed"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model_params = {\n",
        "    'Company': {\n",
        "        'arima_order': (3, 0, 2),\n",
        "        'sarimax_order': (2, 1, 2),\n",
        "        'seasonal_order': (1, 0, 2, 7)\n",
        "    },\n",
        "    'Region 1': {\n",
        "        'arima_order': (3, 1, 3),\n",
        "        'sarimax_order': (2, 1, 1),\n",
        "        'seasonal_order': (2, 1, 0, 7)\n",
        "    },\n",
        "    'Region 2': {\n",
        "        'arima_order': (3, 1, 3),\n",
        "        'sarimax_order': (0, 1, 2),\n",
        "        'seasonal_order': (2, 1, 0, 7)\n",
        "    },\n",
        "    'Region 3': {\n",
        "        'arima_order': (3, 0, 2),\n",
        "        'sarimax_order': (0, 1, 1),\n",
        "        'seasonal_order': (2, 1, 0, 7)\n",
        "    },\n",
        "    'Region 4': {\n",
        "        'arima_order': (1, 1, 1),\n",
        "        'sarimax_order': (2, 1, 2),\n",
        "        'seasonal_order': (1, 0, 2, 7)\n",
        "    }\n",
        "}\n",
        "\n",
        "print(model_params)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HGNAYGihVWKw",
        "outputId": "efea1aba-e870-404a-c8a8-3514e35d27eb"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'Company': {'arima_order': (3, 0, 2), 'sarimax_order': (2, 1, 2), 'seasonal_order': (1, 0, 2, 7)}, 'Region 1': {'arima_order': (3, 1, 3), 'sarimax_order': (2, 1, 1), 'seasonal_order': (2, 1, 0, 7)}, 'Region 2': {'arima_order': (3, 1, 3), 'sarimax_order': (0, 1, 2), 'seasonal_order': (2, 1, 0, 7)}, 'Region 3': {'arima_order': (3, 0, 2), 'sarimax_order': (0, 1, 1), 'seasonal_order': (2, 1, 0, 7)}, 'Region 4': {'arima_order': (1, 1, 1), 'sarimax_order': (2, 1, 2), 'seasonal_order': (1, 0, 2, 7)}}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Recursive Linear Regression Forecasting**"
      ],
      "metadata": {
        "id": "qhua-KxYTYMS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Function performs recursive forecasting using Linear Regression or XGBoost.\n",
        "\n",
        "#Parameters:\n",
        "  # df_train: Processed training data (pandas dataframe) with target_col\n",
        "  # df_inference: Processed inference data (no target_col required)\n",
        "  # model_type: 'lr' for Linear Regression or 'xgb' for XGBoost\n",
        "  # target_col: Target variable name (default='Sales')\n",
        "\n",
        "#Returns:\n",
        "  # Pandad DataFrame with forecasted Sales for the inference period\n",
        "\n",
        "\n",
        "def recursive_forecast(df_train, df_inference, model='lr', target_col='Sales'):\n",
        "    df_train = df_train.copy()\n",
        "    df_inference = df_inference.copy()\n",
        "\n",
        "    df_train.index.freq = 'D'\n",
        "    df_inference.index.freq = 'D'\n",
        "\n",
        "    y_train = df_train[target_col]\n",
        "    X_train = df_train.drop(columns=[target_col])\n",
        "    X_inference = df_inference.drop(columns=[target_col]) if target_col in df_inference.columns else df_inference # if inference used for testing\n",
        "\n",
        "    max_lag = 31\n",
        "    window_features = RollingFeatures(\n",
        "        stats=['mean', 'mean', 'mean'],\n",
        "        window_sizes=[7, 14, 31]\n",
        "    )\n",
        "\n",
        "    if model == 'lr':\n",
        "        forecaster = ForecasterRecursive(\n",
        "            regressor=LinearRegression(),\n",
        "            lags=[1, 2, 3, 7, 31],\n",
        "            window_features=window_features\n",
        "        )\n",
        "    elif model == 'xgb':\n",
        "        forecaster = ForecasterRecursive(\n",
        "            regressor=XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),\n",
        "            lags=[1, 2, 3, 7, 31]\n",
        "        )\n",
        "    else:\n",
        "        raise ValueError(\"Invalid model type. Choose 'lr' or 'xgb'.\")\n",
        "\n",
        "    forecaster.fit(y=y_train, exog=X_train)\n",
        "    y_pred = forecaster.predict(steps=len(df_inference), exog=X_inference, last_window=y_train[-max_lag:])\n",
        "\n",
        "    df_forecast = df_inference.copy()\n",
        "    df_forecast['Sales'] = y_pred.values\n",
        "\n",
        "    return df_forecast\n"
      ],
      "metadata": {
        "id": "j5-atW0VWfpe"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **ARIMA Forecasting**"
      ],
      "metadata": {
        "id": "eSIB6PxkTzX1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# The arima_forecast function will\n",
        "  # Train the model on given data using pre-computed best p,d,q order\n",
        "  # Use model to forecast m steps in the future\n",
        "\n",
        "\n",
        "def arima_forecast(df_train, m_steps, arima_order=(1, 1, 1), target_col='Sales'):\n",
        "    df_train = df_train.copy()\n",
        "    df_train.index.freq = 'D'\n",
        "\n",
        "    # Fit ARIMA model\n",
        "    model = ARIMA(df_train[target_col], order=arima_order)\n",
        "    model_fit = model.fit()\n",
        "\n",
        "    # Forecast for the inference period\n",
        "    forecast = model_fit.forecast(steps=m_steps)\n",
        "\n",
        "    # Prepare output DataFrame\n",
        "    future_index = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=m_steps, freq='D')\n",
        "    df_forecast = pd.DataFrame({target_col: forecast}, index=future_index)\n",
        "\n",
        "    return df_forecast\n",
        "\n"
      ],
      "metadata": {
        "id": "-nzpSXKrgywU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **SARIMAX Forecasting**"
      ],
      "metadata": {
        "id": "vcen0G6fUEx-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# The sarima_forecast function will\n",
        "  # Train the model on given data using pre-computed best p,d,q,P,D,Q,s order\n",
        "  # Use model to forecast m steps in the future\n",
        "\n",
        "def sarimax_forecast(df_train, m_steps, exog_train, exog_pred,\n",
        "                     order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), target_col='Sales'):\n",
        "\n",
        "    df_train = df_train.copy()\n",
        "    exog_train = exog_train.copy()\n",
        "    exog_pred = exog_pred.copy()\n",
        "\n",
        "    df_train.index.freq = 'D'\n",
        "    exog_train.index.freq = 'D'\n",
        "    exog_pred.index.freq = 'D'\n",
        "\n",
        "    # Fit SARIMAX on full training data\n",
        "    model = SARIMAX(df_train[target_col],\n",
        "                    order=order,\n",
        "                    seasonal_order=seasonal_order,\n",
        "                    exog=exog_train)\n",
        "\n",
        "    model_fit = model.fit(disp=False)\n",
        "\n",
        "    # Forecast for the inference period\n",
        "    forecast = model_fit.forecast(steps=m_steps, exog=exog_pred)\n",
        "\n",
        "    # Prepare output DataFrame\n",
        "    future_index = pd.date_range(start=df_train.index[-1] + pd.Timedelta(days=1), periods=m_steps, freq='D')\n",
        "    df_forecast = pd.DataFrame({target_col: forecast}, index=future_index)\n",
        "\n",
        "    return df_forecast"
      ],
      "metadata": {
        "id": "TDtKsK_fH4Vm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Prophet Forecasting**"
      ],
      "metadata": {
        "id": "_VnB9rplUVrh"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def prophet_forecast(ts_data, test_size, m_steps, exog, exog_pred):\n",
        "    warnings.filterwarnings(\"ignore\")\n",
        "\n",
        "    # Step 1: Creating dataframe for Prophet\n",
        "    df = ts_data.reset_index()\n",
        "    df.columns = ['ds', 'y']\n",
        "    df = pd.concat([df, exog.reset_index(drop=True)], axis=1)\n",
        "\n",
        "    # Split into train and test\n",
        "    train_df = df[:-test_size]\n",
        "    test_df = df[-test_size:]\n",
        "\n",
        "    # Step 2: Fit Prophet Model\n",
        "    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=1.25, seasonality_mode='additive')\n",
        "    model.add_regressor('Holiday')\n",
        "    model.add_regressor('Discounted Stores')\n",
        "    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)\n",
        "    #model.add_seasonality(name='14day', period=14, fourier_order=5)\n",
        "    model.fit(train_df)\n",
        "\n",
        "    # Step 3: Create future dataframe for test\n",
        "    future_test = test_df[['ds', 'Holiday', 'Discounted Stores']]\n",
        "    forecast_test = model.predict(future_test)\n",
        "\n",
        "    # Compute MAPE\n",
        "    test_mape = mape(test_df['y'], forecast_test['yhat'])\n",
        "    print(f\"\\nProphet MAPE on test split: {test_mape:.4f}\")\n",
        "\n",
        "    # Step 4: Plot Test Prediction vs Actual\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.plot(test_df['ds'], test_df['y'], label='Actual Test Data')\n",
        "    plt.plot(test_df['ds'], forecast_test['yhat'], label='Test Prediction', linestyle='--')\n",
        "    plt.title('Prophet Forecast vs Actual on Test Set')\n",
        "    plt.xticks(rotation=45)\n",
        "    plt.legend()\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    # Step 5: Forecast Future m Steps using full data\n",
        "    full_df = df.copy()\n",
        "    model_full = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=1.25, seasonality_mode='additive')\n",
        "    model_full.add_regressor('Holiday')\n",
        "    model_full.add_regressor('Discounted Stores')\n",
        "    model_full.add_seasonality(name='monthly', period=30.5, fourier_order=5)\n",
        "    #model_full.add_seasonality(name='14day', period=14, fourier_order=5)\n",
        "    model_full.fit(full_df)\n",
        "\n",
        "    # Prepare future df (including exog_pred)\n",
        "    future_forecast = pd.concat([\n",
        "        full_df[['ds', 'Holiday', 'Discounted Stores']],\n",
        "        exog_pred.reset_index(drop=True)\n",
        "    ], axis=0).tail(m_steps)\n",
        "\n",
        "    future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=m_steps, freq='D')\n",
        "    future_forecast['ds'] = future_dates\n",
        "    forecast_future = model_full.predict(future_forecast)\n",
        "\n",
        "    # Step 6: Plot Full Forecast\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.plot(ts_data.index[-100:], ts_data[-100:], label='Last 100 Data Points')\n",
        "    plt.plot(future_dates, forecast_future['yhat'], label='Forecast from Full Data', linestyle='--')\n",
        "    plt.title('Prophet Forecast for the Next m Steps')\n",
        "    plt.xticks(rotation=45)\n",
        "    plt.legend()\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "\n",
        "    return forecast_future[['ds', 'yhat']]\n"
      ],
      "metadata": {
        "id": "zgFVey7qtLgK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Plot Function**"
      ],
      "metadata": {
        "id": "_QJPbJSok_A9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\"\"\"\n",
        "    Plot historical and forecasted sales for a single model.\n",
        "\n",
        "    Parameters:\n",
        "    - df_train: DataFrame with 'Sales' and datetime index (training data)\n",
        "    - df_forecast: DataFrame with 'Sales' and datetime index (forecasted values)\n",
        "    - model_name: Name of the forecasting model (str)\n",
        "    - inf_label: Optional label for the plot (e.g., 'Company' or 'Region')\n",
        "    \"\"\"\n",
        "\n",
        "def plot_model_forecast(df_train, df_forecast, model_name, inf_label=''):\n",
        "\n",
        "\n",
        "    fig, ax = plt.subplots(figsize=(14, 5))\n",
        "\n",
        "    # Historical data (last 100 days)\n",
        "    ax.plot(df_train.index[-100:], df_train['Sales'][-100:], label='Historical Sales', color='black')\n",
        "\n",
        "    # Forecast\n",
        "    ax.plot(df_forecast.index, df_forecast['Sales'], linestyle='--', label=f'Forecast ({model_name})', color='blue')\n",
        "\n",
        "    ax.set_title(f\"Forecasted Sales for {inf_label} ({model_name})\")\n",
        "    ax.set_xlabel(\"Date\")\n",
        "    ax.set_ylabel(\"Sales\")\n",
        "    ax.legend()\n",
        "    ax.grid(True)\n",
        "    plt.xticks(rotation=45)\n",
        "    plt.tight_layout()\n",
        "\n",
        "    return fig\n"
      ],
      "metadata": {
        "id": "0-YCHQU-U8bv"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}