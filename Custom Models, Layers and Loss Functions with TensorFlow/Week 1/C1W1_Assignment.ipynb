{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "ZL_6GK8qX35J"
   },
   "source": [
    "\n",
    "\n",
    "# Week 1: Multiple Output Models using the Keras Functional API\n",
    "\n",
    "Welcome to the first programming assignment of the course! Your task will be to use the Keras functional API to train a model to predict two outputs. For this lab, you will use the **[Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)** from the **UCI machine learning repository**. It has separate datasets for red wine and white wine.\n",
    "\n",
    "Normally, the wines are classified into one of the quality ratings specified in the attributes. In this exercise, you will combine the two datasets to predict the wine quality and whether the wine is red or white solely from the attributes. \n",
    "\n",
    "You will model wine quality estimations as a regression problem and wine type detection as a binary classification problem.\n",
    "\n",
    "#### Please complete sections that are marked **(TODO)**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "obdcD6urYBY9"
   },
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "t8N3pcTQ5oQI"
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import Dense, Input\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import itertools\n",
    "\n",
    "import utils"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "gQMERzWQYpgm"
   },
   "source": [
    "## Load Dataset\n",
    "\n",
    "\n",
    "You will now load the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) which are **already saved** in your workspace (*Note: For successful grading, please **do not** modify the default string set to the `URI` variable below*).\n",
    "\n",
    "### Pre-process the white wine dataset (TODO)\n",
    "You will add a new column named `is_red` in your dataframe to indicate if the wine is white or red. \n",
    "- In the white wine dataset, you will fill the column `is_red` with  zeros (0)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "2qYAjKXCd4RH",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "325ea195519b7035934c95bb529a062c",
     "grade": false,
     "grade_id": "cell-e5bfa0f152d9a21f",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# URL of the white wine dataset\n",
    "URI = './winequality-white.csv'\n",
    "\n",
    "# load the dataset from the URL\n",
    "white_df = pd.read_csv(URI, sep=\";\")\n",
    "\n",
    "# fill the `is_red` column with zeros.\n",
    "white_df[\"is_red\"] = np.zeros(shape=(white_df.shape[0], 1), dtype=np.int64)\n",
    "\n",
    "# keep only the first of duplicate items\n",
    "white_df = white_df.drop_duplicates(keep='first')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "defe38d6ec58fd31cd67b89e46c4373f",
     "grade": true,
     "grade_id": "cell-30575e713b55fc51",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "# You can click `File -> Open` in the menu above and open the `utils.py` file \n",
    "# in case you want to inspect the unit tests being used for each graded function.\n",
    "\n",
    "utils.test_white_df(white_df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "OQHK0ohBQRCk"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "8.8\n",
      "9.1\n"
     ]
    }
   ],
   "source": [
    "print(white_df.alcohol[0])\n",
    "print(white_df.alcohol[100])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# 8.8\n",
    "# 9.1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Pre-process the red wine dataset (TODO)\n",
    "- In the red wine dataset, you will fill in the column `is_red` with ones (1)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "8y3QxKwBed8v",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "12e0963d15be33b01b4e6ebc8945e51e",
     "grade": false,
     "grade_id": "cell-e47a40f306593274",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# URL of the red wine dataset\n",
    "URI = './winequality-red.csv'\n",
    "\n",
    "# load the dataset from the URL\n",
    "red_df = pd.read_csv(URI, sep=\";\")\n",
    "\n",
    "# fill the `is_red` column with ones.\n",
    "red_df[\"is_red\"] = np.ones(shape=(red_df.shape[0], 1), dtype=np.int64)\n",
    "\n",
    "# keep only the first of duplicate items\n",
    "red_df = red_df.drop_duplicates(keep='first')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "d8e0c91b0fd668b63ba74a8f2f958b59",
     "grade": true,
     "grade_id": "cell-2a75937adcc0c25b",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_red_df(red_df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "zsB3LUzNQpo_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9.4\n",
      "10.2\n"
     ]
    }
   ],
   "source": [
    "print(red_df.alcohol[0])\n",
    "print(red_df.alcohol[100])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# 9.4\n",
    "# 10.2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "2G8B-NYuM6-f"
   },
   "source": [
    "### Concatenate the datasets\n",
    "\n",
    "Next, concatenate the red and white wine dataframes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "YpQrOjJbfN3m"
   },
   "outputs": [],
   "source": [
    "df = pd.concat([red_df, white_df], ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Se2dTmThQyjb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9.4\n",
      "9.5\n"
     ]
    }
   ],
   "source": [
    "print(df.alcohol[0])\n",
    "print(df.alcohol[100])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# 9.4\n",
    "# 9.5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In a real-world scenario, you should shuffle the data. For this assignment however, **you are not** going to do that because the grader needs to test with deterministic data. If you want the code to do it **after** you've gotten your grade for this notebook, we left the commented line below for reference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "wx6y3rPpQv4k"
   },
   "outputs": [],
   "source": [
    "# df = df.iloc[np.random.permutation(len(df))]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "-EqIcbg5M_n1"
   },
   "source": [
    "This will chart the quality of the wines."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "IsvK0-Sgy17C"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['quality'].hist(bins=20);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Nut1rmYLzf-p"
   },
   "source": [
    "### Imbalanced data (TODO)\n",
    "You can see from the plot above that the wine quality dataset is imbalanced. \n",
    "- Since there are very few observations with quality equal to 3, 4, 8 and 9, you can drop these observations from your dataset. \n",
    "- You can do this by removing data belonging to all classes except those > 4 and < 8."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "doH9_-gnf3sz",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "d9ba9fc3a3ca02ccc567be33652b80fe",
     "grade": false,
     "grade_id": "cell-6a3e9db696f6827b",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# get data with wine quality greater than 4 and less than 8\n",
    "df = df[(df['quality'] > 4) & (df['quality'] < 8 )]\n",
    "\n",
    "# reset index and drop the old one\n",
    "df = df.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "281e1d86a4803560ed5892cd7eda4c01",
     "grade": true,
     "grade_id": "cell-aed3da719d4682c7",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_df_drop(df)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "xNR1iAlMRPXO"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9.4\n",
      "10.9\n"
     ]
    }
   ],
   "source": [
    "print(df.alcohol[0])\n",
    "print(df.alcohol[100])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# 9.4\n",
    "# 10.9"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "cwhuRpnVRTzG"
   },
   "source": [
    "You can plot again to see the new range of data and quality"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "857ygzZiLgGg"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['quality'].hist(bins=20);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "n3k0vqSsp84t"
   },
   "source": [
    "### Train Test Split (TODO)\n",
    "\n",
    "Next, you can split the datasets into training, test and validation datasets.\n",
    "- The data frame should be split 80:20 into `train` and `test` sets.\n",
    "- The resulting `train` should then be split 80:20 into `train` and `val` sets.\n",
    "- The `train_test_split` parameter `test_size` takes a float value that ranges between 0. and 1, and represents the proportion of the dataset that is allocated to the test set.  The rest of the data is allocated to the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "PAVIf2-fgRVY",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "7f5738f4fb51d65adc9a8acbdf2b9970",
     "grade": false,
     "grade_id": "cell-91946cadf745206b",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# Please do not change the random_state parameter. This is needed for grading.\n",
    "\n",
    "# split df into 80:20 train and test sets\n",
    "train, test = train_test_split(df, test_size=0.2, random_state = 1)\n",
    "                               \n",
    "# split train into 80:20 train and val sets\n",
    "train, val = train_test_split(train, test_size=0.2, random_state = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "editable": false,
    "id": "57h9LcEzRWpk",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "42adbe9e66efac7c7a5f8cd73ac92f22",
     "grade": true,
     "grade_id": "cell-64b8b38cd0b965f6",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_data_sizes(train.size, test.size, val.size)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "RwTNu4KFqG-K"
   },
   "source": [
    "Here's where you can explore the training stats. You can pop the labels 'is_red' and 'quality' from the data as these will be used as the labels\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Y_afyhhHM6WQ"
   },
   "outputs": [],
   "source": [
    "train_stats = train.describe()\n",
    "train_stats.pop('is_red')\n",
    "train_stats.pop('quality')\n",
    "train_stats = train_stats.transpose()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "ahvbYm4fNqSt"
   },
   "source": [
    "Explore the training stats!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "n_gAtPjZ0otF"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>count</th>\n",
       "      <th>mean</th>\n",
       "      <th>std</th>\n",
       "      <th>min</th>\n",
       "      <th>25%</th>\n",
       "      <th>50%</th>\n",
       "      <th>75%</th>\n",
       "      <th>max</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>fixed acidity</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>7.221616</td>\n",
       "      <td>1.325297</td>\n",
       "      <td>3.80000</td>\n",
       "      <td>6.40000</td>\n",
       "      <td>7.00000</td>\n",
       "      <td>7.7000</td>\n",
       "      <td>15.60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>volatile acidity</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>0.338929</td>\n",
       "      <td>0.162476</td>\n",
       "      <td>0.08000</td>\n",
       "      <td>0.23000</td>\n",
       "      <td>0.29000</td>\n",
       "      <td>0.4000</td>\n",
       "      <td>1.24000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>citric acid</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>0.321569</td>\n",
       "      <td>0.147970</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.25000</td>\n",
       "      <td>0.31000</td>\n",
       "      <td>0.4000</td>\n",
       "      <td>1.66000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>residual sugar</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>5.155911</td>\n",
       "      <td>4.639632</td>\n",
       "      <td>0.60000</td>\n",
       "      <td>1.80000</td>\n",
       "      <td>2.80000</td>\n",
       "      <td>7.6500</td>\n",
       "      <td>65.80000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>chlorides</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>0.056976</td>\n",
       "      <td>0.036802</td>\n",
       "      <td>0.01200</td>\n",
       "      <td>0.03800</td>\n",
       "      <td>0.04700</td>\n",
       "      <td>0.0660</td>\n",
       "      <td>0.61100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>30.388590</td>\n",
       "      <td>17.236784</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>17.00000</td>\n",
       "      <td>28.00000</td>\n",
       "      <td>41.0000</td>\n",
       "      <td>131.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>115.062282</td>\n",
       "      <td>56.706617</td>\n",
       "      <td>6.00000</td>\n",
       "      <td>75.00000</td>\n",
       "      <td>117.00000</td>\n",
       "      <td>156.0000</td>\n",
       "      <td>344.00000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>density</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>0.994633</td>\n",
       "      <td>0.003005</td>\n",
       "      <td>0.98711</td>\n",
       "      <td>0.99232</td>\n",
       "      <td>0.99481</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>1.03898</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pH</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>3.223201</td>\n",
       "      <td>0.161272</td>\n",
       "      <td>2.72000</td>\n",
       "      <td>3.11000</td>\n",
       "      <td>3.21000</td>\n",
       "      <td>3.3300</td>\n",
       "      <td>4.01000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sulphates</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>0.534051</td>\n",
       "      <td>0.149149</td>\n",
       "      <td>0.22000</td>\n",
       "      <td>0.43000</td>\n",
       "      <td>0.51000</td>\n",
       "      <td>0.6000</td>\n",
       "      <td>1.95000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>alcohol</th>\n",
       "      <td>3155.0</td>\n",
       "      <td>10.504466</td>\n",
       "      <td>1.154654</td>\n",
       "      <td>8.50000</td>\n",
       "      <td>9.50000</td>\n",
       "      <td>10.30000</td>\n",
       "      <td>11.3000</td>\n",
       "      <td>14.00000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       count        mean        std      min       25%  \\\n",
       "fixed acidity         3155.0    7.221616   1.325297  3.80000   6.40000   \n",
       "volatile acidity      3155.0    0.338929   0.162476  0.08000   0.23000   \n",
       "citric acid           3155.0    0.321569   0.147970  0.00000   0.25000   \n",
       "residual sugar        3155.0    5.155911   4.639632  0.60000   1.80000   \n",
       "chlorides             3155.0    0.056976   0.036802  0.01200   0.03800   \n",
       "free sulfur dioxide   3155.0   30.388590  17.236784  1.00000  17.00000   \n",
       "total sulfur dioxide  3155.0  115.062282  56.706617  6.00000  75.00000   \n",
       "density               3155.0    0.994633   0.003005  0.98711   0.99232   \n",
       "pH                    3155.0    3.223201   0.161272  2.72000   3.11000   \n",
       "sulphates             3155.0    0.534051   0.149149  0.22000   0.43000   \n",
       "alcohol               3155.0   10.504466   1.154654  8.50000   9.50000   \n",
       "\n",
       "                            50%       75%        max  \n",
       "fixed acidity           7.00000    7.7000   15.60000  \n",
       "volatile acidity        0.29000    0.4000    1.24000  \n",
       "citric acid             0.31000    0.4000    1.66000  \n",
       "residual sugar          2.80000    7.6500   65.80000  \n",
       "chlorides               0.04700    0.0660    0.61100  \n",
       "free sulfur dioxide    28.00000   41.0000  131.00000  \n",
       "total sulfur dioxide  117.00000  156.0000  344.00000  \n",
       "density                 0.99481    0.9968    1.03898  \n",
       "pH                      3.21000    3.3300    4.01000  \n",
       "sulphates               0.51000    0.6000    1.95000  \n",
       "alcohol                10.30000   11.3000   14.00000  "
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_stats"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "bGPvt9jir_HC"
   },
   "source": [
    "### Get the labels (TODO)\n",
    "\n",
    "The features and labels are currently in the same dataframe.\n",
    "- You will want to store the label columns `is_red` and `quality` separately from the feature columns.  \n",
    "- The following function, `format_output`, gets these two columns from the dataframe (it's given to you).\n",
    "- `format_output` also formats the data into numpy arrays. \n",
    "- Please use the `format_output` and apply it to the `train`, `val` and `test` sets to get dataframes for the labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Z_fs14XQqZVP"
   },
   "outputs": [],
   "source": [
    "def format_output(data):\n",
    "    is_red = data.pop('is_red')\n",
    "    is_red = np.array(is_red)\n",
    "    quality = data.pop('quality')\n",
    "    quality = np.array(quality)\n",
    "    return (quality, is_red)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "8L3ZZe1fQicm",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "7a86809e54895a816434c48dc903f55d",
     "grade": false,
     "grade_id": "cell-5c30fa2c2a354b0f",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# format the output of the train set\n",
    "train_Y = format_output(train)\n",
    "\n",
    "# format the output of the val set\n",
    "val_Y = format_output(val)\n",
    "    \n",
    "# format the output of the test set\n",
    "test_Y = format_output(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "359cabbafaed14ec9bbc1e57a7b6f32c",
     "grade": true,
     "grade_id": "cell-4977d8befb80f56b",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_format_output(df, train_Y, val_Y, test_Y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notice that after you get the labels, the `train`, `val` and `test` dataframes no longer contain the label columns, and contain just the feature columns.\n",
    "- This is because you used `.pop` in the `format_output` function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
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
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>225</th>\n",
       "      <td>7.5</td>\n",
       "      <td>0.65</td>\n",
       "      <td>0.18</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.088</td>\n",
       "      <td>27.0</td>\n",
       "      <td>94.0</td>\n",
       "      <td>0.99915</td>\n",
       "      <td>3.38</td>\n",
       "      <td>0.77</td>\n",
       "      <td>9.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3557</th>\n",
       "      <td>6.3</td>\n",
       "      <td>0.27</td>\n",
       "      <td>0.29</td>\n",
       "      <td>12.2</td>\n",
       "      <td>0.044</td>\n",
       "      <td>59.0</td>\n",
       "      <td>196.0</td>\n",
       "      <td>0.99782</td>\n",
       "      <td>3.14</td>\n",
       "      <td>0.40</td>\n",
       "      <td>8.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3825</th>\n",
       "      <td>8.8</td>\n",
       "      <td>0.27</td>\n",
       "      <td>0.25</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.024</td>\n",
       "      <td>52.0</td>\n",
       "      <td>99.0</td>\n",
       "      <td>0.99250</td>\n",
       "      <td>2.87</td>\n",
       "      <td>0.49</td>\n",
       "      <td>11.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1740</th>\n",
       "      <td>6.4</td>\n",
       "      <td>0.45</td>\n",
       "      <td>0.07</td>\n",
       "      <td>1.1</td>\n",
       "      <td>0.030</td>\n",
       "      <td>10.0</td>\n",
       "      <td>131.0</td>\n",
       "      <td>0.99050</td>\n",
       "      <td>2.97</td>\n",
       "      <td>0.28</td>\n",
       "      <td>10.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1221</th>\n",
       "      <td>7.2</td>\n",
       "      <td>0.53</td>\n",
       "      <td>0.13</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0.058</td>\n",
       "      <td>18.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>0.99573</td>\n",
       "      <td>3.21</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.9</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "225             7.5              0.65         0.18             7.0      0.088   \n",
       "3557            6.3              0.27         0.29            12.2      0.044   \n",
       "3825            8.8              0.27         0.25             5.0      0.024   \n",
       "1740            6.4              0.45         0.07             1.1      0.030   \n",
       "1221            7.2              0.53         0.13             2.0      0.058   \n",
       "\n",
       "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "225                  27.0                  94.0  0.99915  3.38       0.77   \n",
       "3557                 59.0                 196.0  0.99782  3.14       0.40   \n",
       "3825                 52.0                  99.0  0.99250  2.87       0.49   \n",
       "1740                 10.0                 131.0  0.99050  2.97       0.28   \n",
       "1221                 18.0                  22.0  0.99573  3.21       0.68   \n",
       "\n",
       "      alcohol  \n",
       "225       9.4  \n",
       "3557      8.8  \n",
       "3825     11.4  \n",
       "1740     10.8  \n",
       "1221      9.9  "
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "hEdbrruAsN1D"
   },
   "source": [
    "### Normalize the data (TODO)\n",
    "\n",
    "Next, you can normalize the data, x, using the formula:\n",
    "$$x_{norm} = \\frac{x - \\mu}{\\sigma}$$\n",
    "- The `norm` function is defined for you.\n",
    "- Please apply the `norm` function to normalize the dataframes that contains the feature columns of `train`, `val` and `test` sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "WWiZPAHCLjUs"
   },
   "outputs": [],
   "source": [
    "def norm(x):\n",
    "    return (x - train_stats['mean']) / train_stats['std']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "JEaOi2I2Lk69",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "6bc0cdcb563d192f271067aa3373ff32",
     "grade": false,
     "grade_id": "cell-d8416d975c371095",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "# normalize the train set\n",
    "norm_train_X = norm(train)\n",
    "    \n",
    "# normalize the val set\n",
    "norm_val_X = norm(val)\n",
    "    \n",
    "# normalize the test set\n",
    "norm_test_X = norm(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "4f567db45bf40191601780379cc100b8",
     "grade": true,
     "grade_id": "cell-97fad979d157529b",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "hzykDwQhsaPO"
   },
   "source": [
    "## Define the Model (TODO)\n",
    "\n",
    "Define the model using the functional API. The base model will be 2 `Dense` layers of 128 neurons each, and have the `'relu'` activation.\n",
    "- Check out the documentation for [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "Rhcns3oTFkM6",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "74b031247e569526552bf13a034a1c07",
     "grade": false,
     "grade_id": "cell-73fceedad1fe351c",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "def base_model(inputs):\n",
    "    \n",
    "    # connect a Dense layer with 128 neurons and a relu activation\n",
    "    x = Dense(units=128, activation='relu')(inputs)\n",
    "    \n",
    "    # connect another Dense layer with 128 neurons and a relu activation\n",
    "    x = Dense(units=128, activation='relu')(x)\n",
    "    return x\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "9255924b3def80f679616e4c851a43e1",
     "grade": true,
     "grade_id": "cell-54f742a133353d75",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_base_model(base_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "xem_fcVws6Kz"
   },
   "source": [
    "# Define output layers of the model (TODO)\n",
    "\n",
    "You will add output layers to the base model. \n",
    "- The model will need two outputs.\n",
    "\n",
    "One output layer will predict wine quality, which is a numeric value.\n",
    "- Define a `Dense` layer with 1 neuron.\n",
    "- Since this is a regression output, the activation can be left as its default value `None`.\n",
    "\n",
    "The other output layer will predict the wine type, which is either red `1` or not red `0` (white).\n",
    "- Define a `Dense` layer with 1 neuron.\n",
    "- Since there are two possible categories, you can use a sigmoid activation for binary classification.\n",
    "\n",
    "Define the `Model`\n",
    "- Define the `Model` object, and set the following parameters:\n",
    "  - `inputs`: pass in the inputs to the model as a list.\n",
    "  - `outputs`: pass in a list of the outputs that you just defined: wine quality, then wine type.\n",
    "  - **Note**: please list the wine quality before wine type in the outputs, as this will affect the calculated loss if you choose the other order."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "n5UGF8PMVLPt",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "76d35b90d20cdcbb22986cd8211057de",
     "grade": false,
     "grade_id": "cell-19e285f482f021fb",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "def final_model(inputs):\n",
    "    \n",
    "    # get the base model\n",
    "    x = base_model(inputs)\n",
    "\n",
    "    # connect the output Dense layer for regression\n",
    "    wine_quality = Dense(units='1', name='wine_quality')(x)\n",
    "\n",
    "    # connect the output Dense layer for classification. this will use a sigmoid activation.\n",
    "    wine_type = Dense(units='1', activation='sigmoid', name='wine_type')(x)\n",
    "\n",
    "    # define the model using the input and output layers\n",
    "    model = Model(inputs=inputs, outputs=[wine_quality, wine_type])\n",
    "\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "89cbf89d8ab5e2e59ecf7f63f517520a",
     "grade": true,
     "grade_id": "cell-40d050f855c817d1",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_final_model(final_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "5R0BMTsltZyu"
   },
   "source": [
    "## Compiling the Model\n",
    "\n",
    "Next, compile the model. When setting the loss parameter of `model.compile`, you're setting the loss for each of the two outputs (wine quality and wine type).\n",
    "\n",
    "To set more than one loss, use a dictionary of key-value pairs.\n",
    "- You can look at the docs for the losses [here](https://www.tensorflow.org/api_docs/python/tf/keras/losses#functions).\n",
    "    - **Note**: For the desired spelling, please look at the \"Functions\" section of the documentation and not the \"classes\" section on that same page.\n",
    "- wine_type: Since you will be performing binary classification on wine type, you should use the binary crossentropy loss function for it.  Please pass this in as a string.  \n",
    "  - **Hint**, this should be all lowercase.  In the documentation, you'll see this under the \"Functions\" section, not the \"Classes\" section.\n",
    "- wine_quality: since this is a regression output, use the mean squared error.  Please pass it in as a string, all lowercase.\n",
    "  - **Hint**: You may notice that there are two aliases for mean squared error.  Please use the shorter name.\n",
    "\n",
    "\n",
    "You will also set the metric for each of the two outputs.  Again, to set metrics for two or more outputs, use a dictionary with key value pairs.\n",
    "- The metrics documentation is linked [here](https://www.tensorflow.org/api_docs/python/tf/keras/metrics).\n",
    "- For the wine type, please set it to accuracy as a string, all lowercase.\n",
    "- For wine quality, please use the root mean squared error.  Instead of a string, you'll set it to an instance of the class [RootMeanSquaredError](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError), which belongs to the tf.keras.metrics module.\n",
    "\n",
    "**Note**: If you see the error message \n",
    ">Exception: wine quality loss function is incorrect.\n",
    "\n",
    "- Please also check your other losses and metrics, as the error may be caused by the other three key-value pairs and not the wine quality loss."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "LK11duUbUjmh",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "22f45067ca69eb2ccadb43874dbcc27b",
     "grade": false,
     "grade_id": "cell-81afdc4dcca51d5e",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "inputs = tf.keras.layers.Input(shape=(11,))\n",
    "rms = tf.keras.optimizers.RMSprop(lr=0.0001)\n",
    "model = final_model(inputs)\n",
    "\n",
    "model.compile(optimizer=rms, \n",
    "              loss = {'wine_type' : 'binary_crossentropy',\n",
    "                      'wine_quality' : 'mse'\n",
    "                     },\n",
    "              metrics = {'wine_type' : 'accuracy',\n",
    "                         'wine_quality': tf.keras.metrics.RootMeanSquaredError()\n",
    "                       }\n",
    "             )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "88e02238ea5e456ff65e835cc8158054",
     "grade": true,
     "grade_id": "cell-2eeeba02391c4632",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_model_compile(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "90MpAMpWuKm-"
   },
   "source": [
    "## Training the Model (TODO)\n",
    "\n",
    "Fit the model to the training inputs and outputs. \n",
    "- Check the documentation for [model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).\n",
    "- Remember to use the normalized training set as inputs. \n",
    "- For the validation data, please use the normalized validation set.\n",
    "\n",
    "**Important: Please do not increase the number of epochs below. This is to avoid the grader from timing out. You can increase it once you have submitted your work.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "deletable": false,
    "id": "_eiZkle4XwiY",
    "nbgrader": {
     "cell_type": "code",
     "checksum": "d1a4565296017a0611c6f2de675f96cf",
     "grade": false,
     "grade_id": "cell-0bb56262896f6680",
     "locked": false,
     "schema_version": 3,
     "solution": true,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 3155 samples, validate on 789 samples\n",
      "Epoch 1/40\n",
      "3155/3155 [==============================] - 1s 357us/sample - loss: 22.0295 - wine_quality_loss: 21.3532 - wine_type_loss: 0.6444 - wine_quality_root_mean_squared_error: 4.6244 - wine_type_accuracy: 0.7084 - val_loss: 14.5611 - val_wine_quality_loss: 13.9847 - val_wine_type_loss: 0.6210 - val_wine_quality_root_mean_squared_error: 3.7338 - val_wine_type_accuracy: 0.7338\n",
      "Epoch 2/40\n",
      "3155/3155 [==============================] - 0s 129us/sample - loss: 9.4527 - wine_quality_loss: 8.8630 - wine_type_loss: 0.5801 - wine_quality_root_mean_squared_error: 2.9786 - wine_type_accuracy: 0.7442 - val_loss: 5.5179 - val_wine_quality_loss: 5.0715 - val_wine_type_loss: 0.5415 - val_wine_quality_root_mean_squared_error: 2.2310 - val_wine_type_accuracy: 0.7338\n",
      "Epoch 3/40\n",
      "3155/3155 [==============================] - 0s 122us/sample - loss: 3.9198 - wine_quality_loss: 3.4409 - wine_type_loss: 0.4779 - wine_quality_root_mean_squared_error: 1.8551 - wine_type_accuracy: 0.7686 - val_loss: 2.7842 - val_wine_quality_loss: 2.4308 - val_wine_type_loss: 0.4231 - val_wine_quality_root_mean_squared_error: 1.5369 - val_wine_type_accuracy: 0.8302\n",
      "Epoch 4/40\n",
      "3155/3155 [==============================] - 0s 120us/sample - loss: 2.5471 - wine_quality_loss: 2.1817 - wine_type_loss: 0.3634 - wine_quality_root_mean_squared_error: 1.4777 - wine_type_accuracy: 0.8897 - val_loss: 2.2201 - val_wine_quality_loss: 1.9376 - val_wine_type_loss: 0.3211 - val_wine_quality_root_mean_squared_error: 1.3783 - val_wine_type_accuracy: 0.9176\n",
      "Epoch 5/40\n",
      "3155/3155 [==============================] - 0s 103us/sample - loss: 2.1033 - wine_quality_loss: 1.8257 - wine_type_loss: 0.2742 - wine_quality_root_mean_squared_error: 1.3523 - wine_type_accuracy: 0.9493 - val_loss: 1.9067 - val_wine_quality_loss: 1.6863 - val_wine_type_loss: 0.2443 - val_wine_quality_root_mean_squared_error: 1.2895 - val_wine_type_accuracy: 0.9645\n",
      "Epoch 6/40\n",
      "3155/3155 [==============================] - 0s 101us/sample - loss: 1.8226 - wine_quality_loss: 1.6112 - wine_type_loss: 0.2106 - wine_quality_root_mean_squared_error: 1.2696 - wine_type_accuracy: 0.9699 - val_loss: 1.6641 - val_wine_quality_loss: 1.4890 - val_wine_type_loss: 0.1880 - val_wine_quality_root_mean_squared_error: 1.2150 - val_wine_type_accuracy: 0.9823\n",
      "Epoch 7/40\n",
      "3155/3155 [==============================] - 0s 118us/sample - loss: 1.6089 - wine_quality_loss: 1.4439 - wine_type_loss: 0.1641 - wine_quality_root_mean_squared_error: 1.2020 - wine_type_accuracy: 0.9810 - val_loss: 1.4840 - val_wine_quality_loss: 1.3440 - val_wine_type_loss: 0.1471 - val_wine_quality_root_mean_squared_error: 1.1563 - val_wine_type_accuracy: 0.9873\n",
      "Epoch 8/40\n",
      "3155/3155 [==============================] - 0s 101us/sample - loss: 1.4487 - wine_quality_loss: 1.3162 - wine_type_loss: 0.1308 - wine_quality_root_mean_squared_error: 1.1479 - wine_type_accuracy: 0.9867 - val_loss: 1.3443 - val_wine_quality_loss: 1.2306 - val_wine_type_loss: 0.1178 - val_wine_quality_root_mean_squared_error: 1.1074 - val_wine_type_accuracy: 0.9886\n",
      "Epoch 9/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 1.3121 - wine_quality_loss: 1.2066 - wine_type_loss: 0.1068 - wine_quality_root_mean_squared_error: 1.0979 - wine_type_accuracy: 0.9892 - val_loss: 1.2160 - val_wine_quality_loss: 1.1205 - val_wine_type_loss: 0.0974 - val_wine_quality_root_mean_squared_error: 1.0577 - val_wine_type_accuracy: 0.9899\n",
      "Epoch 10/40\n",
      "3155/3155 [==============================] - 0s 104us/sample - loss: 1.1933 - wine_quality_loss: 1.1025 - wine_type_loss: 0.0896 - wine_quality_root_mean_squared_error: 1.0505 - wine_type_accuracy: 0.9911 - val_loss: 1.1316 - val_wine_quality_loss: 1.0495 - val_wine_type_loss: 0.0826 - val_wine_quality_root_mean_squared_error: 1.0242 - val_wine_type_accuracy: 0.9911\n",
      "Epoch 11/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 1.1018 - wine_quality_loss: 1.0254 - wine_type_loss: 0.0773 - wine_quality_root_mean_squared_error: 1.0121 - wine_type_accuracy: 0.9911 - val_loss: 1.0487 - val_wine_quality_loss: 0.9759 - val_wine_type_loss: 0.0719 - val_wine_quality_root_mean_squared_error: 0.9883 - val_wine_type_accuracy: 0.9911\n",
      "Epoch 12/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 1.0190 - wine_quality_loss: 0.9511 - wine_type_loss: 0.0681 - wine_quality_root_mean_squared_error: 0.9751 - wine_type_accuracy: 0.9918 - val_loss: 0.9610 - val_wine_quality_loss: 0.8961 - val_wine_type_loss: 0.0639 - val_wine_quality_root_mean_squared_error: 0.9471 - val_wine_type_accuracy: 0.9924\n",
      "Epoch 13/40\n",
      "3155/3155 [==============================] - 0s 117us/sample - loss: 0.9459 - wine_quality_loss: 0.8840 - wine_type_loss: 0.0612 - wine_quality_root_mean_squared_error: 0.9405 - wine_type_accuracy: 0.9918 - val_loss: 0.8960 - val_wine_quality_loss: 0.8369 - val_wine_type_loss: 0.0576 - val_wine_quality_root_mean_squared_error: 0.9155 - val_wine_type_accuracy: 0.9937\n",
      "Epoch 14/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 0.8814 - wine_quality_loss: 0.8265 - wine_type_loss: 0.0565 - wine_quality_root_mean_squared_error: 0.9085 - wine_type_accuracy: 0.9918 - val_loss: 0.8293 - val_wine_quality_loss: 0.7745 - val_wine_type_loss: 0.0532 - val_wine_quality_root_mean_squared_error: 0.8808 - val_wine_type_accuracy: 0.9937\n",
      "Epoch 15/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.8244 - wine_quality_loss: 0.7712 - wine_type_loss: 0.0516 - wine_quality_root_mean_squared_error: 0.8789 - wine_type_accuracy: 0.9921 - val_loss: 0.7936 - val_wine_quality_loss: 0.7427 - val_wine_type_loss: 0.0493 - val_wine_quality_root_mean_squared_error: 0.8626 - val_wine_type_accuracy: 0.9937\n",
      "Epoch 16/40\n",
      "3155/3155 [==============================] - 0s 117us/sample - loss: 0.7731 - wine_quality_loss: 0.7248 - wine_type_loss: 0.0498 - wine_quality_root_mean_squared_error: 0.8512 - wine_type_accuracy: 0.9921 - val_loss: 0.7322 - val_wine_quality_loss: 0.6841 - val_wine_type_loss: 0.0464 - val_wine_quality_root_mean_squared_error: 0.8280 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 17/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 0.7262 - wine_quality_loss: 0.6810 - wine_type_loss: 0.0459 - wine_quality_root_mean_squared_error: 0.8248 - wine_type_accuracy: 0.9924 - val_loss: 0.6917 - val_wine_quality_loss: 0.6457 - val_wine_type_loss: 0.0439 - val_wine_quality_root_mean_squared_error: 0.8047 - val_wine_type_accuracy: 0.9962\n",
      "Epoch 18/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.6852 - wine_quality_loss: 0.6423 - wine_type_loss: 0.0437 - wine_quality_root_mean_squared_error: 0.8010 - wine_type_accuracy: 0.9924 - val_loss: 0.6489 - val_wine_quality_loss: 0.6051 - val_wine_type_loss: 0.0421 - val_wine_quality_root_mean_squared_error: 0.7788 - val_wine_type_accuracy: 0.9962\n",
      "Epoch 19/40\n",
      "3155/3155 [==============================] - 0s 114us/sample - loss: 0.6475 - wine_quality_loss: 0.6054 - wine_type_loss: 0.0418 - wine_quality_root_mean_squared_error: 0.7783 - wine_type_accuracy: 0.9927 - val_loss: 0.6111 - val_wine_quality_loss: 0.5689 - val_wine_type_loss: 0.0403 - val_wine_quality_root_mean_squared_error: 0.7553 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 20/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 0.6147 - wine_quality_loss: 0.5738 - wine_type_loss: 0.0401 - wine_quality_root_mean_squared_error: 0.7580 - wine_type_accuracy: 0.9927 - val_loss: 0.5859 - val_wine_quality_loss: 0.5448 - val_wine_type_loss: 0.0390 - val_wine_quality_root_mean_squared_error: 0.7393 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 21/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 0.5832 - wine_quality_loss: 0.5447 - wine_type_loss: 0.0387 - wine_quality_root_mean_squared_error: 0.7379 - wine_type_accuracy: 0.9927 - val_loss: 0.5654 - val_wine_quality_loss: 0.5254 - val_wine_type_loss: 0.0377 - val_wine_quality_root_mean_squared_error: 0.7262 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 22/40\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3155/3155 [==============================] - 0s 99us/sample - loss: 0.5555 - wine_quality_loss: 0.5180 - wine_type_loss: 0.0374 - wine_quality_root_mean_squared_error: 0.7197 - wine_type_accuracy: 0.9927 - val_loss: 0.5322 - val_wine_quality_loss: 0.4935 - val_wine_type_loss: 0.0367 - val_wine_quality_root_mean_squared_error: 0.7037 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 23/40\n",
      "3155/3155 [==============================] - 0s 114us/sample - loss: 0.5318 - wine_quality_loss: 0.4946 - wine_type_loss: 0.0378 - wine_quality_root_mean_squared_error: 0.7038 - wine_type_accuracy: 0.9933 - val_loss: 0.5050 - val_wine_quality_loss: 0.4674 - val_wine_type_loss: 0.0359 - val_wine_quality_root_mean_squared_error: 0.6847 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 24/40\n",
      "3155/3155 [==============================] - 0s 100us/sample - loss: 0.5078 - wine_quality_loss: 0.4717 - wine_type_loss: 0.0355 - wine_quality_root_mean_squared_error: 0.6872 - wine_type_accuracy: 0.9933 - val_loss: 0.5009 - val_wine_quality_loss: 0.4636 - val_wine_type_loss: 0.0351 - val_wine_quality_root_mean_squared_error: 0.6823 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 25/40\n",
      "3155/3155 [==============================] - 0s 99us/sample - loss: 0.4897 - wine_quality_loss: 0.4551 - wine_type_loss: 0.0347 - wine_quality_root_mean_squared_error: 0.6745 - wine_type_accuracy: 0.9933 - val_loss: 0.4693 - val_wine_quality_loss: 0.4336 - val_wine_type_loss: 0.0344 - val_wine_quality_root_mean_squared_error: 0.6592 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 26/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.4724 - wine_quality_loss: 0.4376 - wine_type_loss: 0.0340 - wine_quality_root_mean_squared_error: 0.6620 - wine_type_accuracy: 0.9933 - val_loss: 0.4558 - val_wine_quality_loss: 0.4205 - val_wine_type_loss: 0.0337 - val_wine_quality_root_mean_squared_error: 0.6494 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 27/40\n",
      "3155/3155 [==============================] - 0s 114us/sample - loss: 0.4565 - wine_quality_loss: 0.4232 - wine_type_loss: 0.0334 - wine_quality_root_mean_squared_error: 0.6504 - wine_type_accuracy: 0.9937 - val_loss: 0.4373 - val_wine_quality_loss: 0.4025 - val_wine_type_loss: 0.0333 - val_wine_quality_root_mean_squared_error: 0.6353 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 28/40\n",
      "3155/3155 [==============================] - 0s 99us/sample - loss: 0.4414 - wine_quality_loss: 0.4078 - wine_type_loss: 0.0329 - wine_quality_root_mean_squared_error: 0.6391 - wine_type_accuracy: 0.9940 - val_loss: 0.4388 - val_wine_quality_loss: 0.4039 - val_wine_type_loss: 0.0330 - val_wine_quality_root_mean_squared_error: 0.6368 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 29/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.4294 - wine_quality_loss: 0.3959 - wine_type_loss: 0.0325 - wine_quality_root_mean_squared_error: 0.6299 - wine_type_accuracy: 0.9940 - val_loss: 0.4137 - val_wine_quality_loss: 0.3799 - val_wine_type_loss: 0.0325 - val_wine_quality_root_mean_squared_error: 0.6172 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 30/40\n",
      "3155/3155 [==============================] - 0s 97us/sample - loss: 0.4185 - wine_quality_loss: 0.3861 - wine_type_loss: 0.0320 - wine_quality_root_mean_squared_error: 0.6217 - wine_type_accuracy: 0.9943 - val_loss: 0.4080 - val_wine_quality_loss: 0.3742 - val_wine_type_loss: 0.0321 - val_wine_quality_root_mean_squared_error: 0.6128 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 31/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.4085 - wine_quality_loss: 0.3775 - wine_type_loss: 0.0316 - wine_quality_root_mean_squared_error: 0.6139 - wine_type_accuracy: 0.9943 - val_loss: 0.4030 - val_wine_quality_loss: 0.3695 - val_wine_type_loss: 0.0319 - val_wine_quality_root_mean_squared_error: 0.6089 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 32/40\n",
      "3155/3155 [==============================] - 0s 113us/sample - loss: 0.4007 - wine_quality_loss: 0.3696 - wine_type_loss: 0.0311 - wine_quality_root_mean_squared_error: 0.6078 - wine_type_accuracy: 0.9943 - val_loss: 0.3896 - val_wine_quality_loss: 0.3567 - val_wine_type_loss: 0.0316 - val_wine_quality_root_mean_squared_error: 0.5980 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 33/40\n",
      "3155/3155 [==============================] - 0s 99us/sample - loss: 0.3931 - wine_quality_loss: 0.3618 - wine_type_loss: 0.0308 - wine_quality_root_mean_squared_error: 0.6018 - wine_type_accuracy: 0.9943 - val_loss: 0.3983 - val_wine_quality_loss: 0.3654 - val_wine_type_loss: 0.0314 - val_wine_quality_root_mean_squared_error: 0.6054 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 34/40\n",
      "3155/3155 [==============================] - 0s 97us/sample - loss: 0.3872 - wine_quality_loss: 0.3567 - wine_type_loss: 0.0305 - wine_quality_root_mean_squared_error: 0.5972 - wine_type_accuracy: 0.9946 - val_loss: 0.3815 - val_wine_quality_loss: 0.3489 - val_wine_type_loss: 0.0311 - val_wine_quality_root_mean_squared_error: 0.5916 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 35/40\n",
      "3155/3155 [==============================] - 0s 97us/sample - loss: 0.3804 - wine_quality_loss: 0.3498 - wine_type_loss: 0.0303 - wine_quality_root_mean_squared_error: 0.5917 - wine_type_accuracy: 0.9946 - val_loss: 0.3782 - val_wine_quality_loss: 0.3460 - val_wine_type_loss: 0.0310 - val_wine_quality_root_mean_squared_error: 0.5889 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 36/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.3749 - wine_quality_loss: 0.3450 - wine_type_loss: 0.0299 - wine_quality_root_mean_squared_error: 0.5872 - wine_type_accuracy: 0.9943 - val_loss: 0.3709 - val_wine_quality_loss: 0.3387 - val_wine_type_loss: 0.0307 - val_wine_quality_root_mean_squared_error: 0.5829 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 37/40\n",
      "3155/3155 [==============================] - 0s 112us/sample - loss: 0.3708 - wine_quality_loss: 0.3406 - wine_type_loss: 0.0297 - wine_quality_root_mean_squared_error: 0.5839 - wine_type_accuracy: 0.9946 - val_loss: 0.3691 - val_wine_quality_loss: 0.3370 - val_wine_type_loss: 0.0307 - val_wine_quality_root_mean_squared_error: 0.5814 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 38/40\n",
      "3155/3155 [==============================] - 0s 99us/sample - loss: 0.3666 - wine_quality_loss: 0.3370 - wine_type_loss: 0.0295 - wine_quality_root_mean_squared_error: 0.5805 - wine_type_accuracy: 0.9949 - val_loss: 0.3683 - val_wine_quality_loss: 0.3367 - val_wine_type_loss: 0.0305 - val_wine_quality_root_mean_squared_error: 0.5809 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 39/40\n",
      "3155/3155 [==============================] - 0s 98us/sample - loss: 0.3627 - wine_quality_loss: 0.3329 - wine_type_loss: 0.0293 - wine_quality_root_mean_squared_error: 0.5774 - wine_type_accuracy: 0.9949 - val_loss: 0.3643 - val_wine_quality_loss: 0.3326 - val_wine_type_loss: 0.0304 - val_wine_quality_root_mean_squared_error: 0.5775 - val_wine_type_accuracy: 0.9949\n",
      "Epoch 40/40\n",
      "3155/3155 [==============================] - 0s 97us/sample - loss: 0.3591 - wine_quality_loss: 0.3302 - wine_type_loss: 0.0290 - wine_quality_root_mean_squared_error: 0.5745 - wine_type_accuracy: 0.9949 - val_loss: 0.3559 - val_wine_quality_loss: 0.3247 - val_wine_type_loss: 0.0303 - val_wine_quality_root_mean_squared_error: 0.5703 - val_wine_type_accuracy: 0.9949\n"
     ]
    }
   ],
   "source": [
    "# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.\n",
    "# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.\n",
    "\n",
    "\n",
    "\n",
    "history = model.fit(norm_train_X, train_Y,\n",
    "                    epochs = 40, validation_data=(norm_val_X, val_Y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {
    "deletable": false,
    "editable": false,
    "nbgrader": {
     "cell_type": "code",
     "checksum": "fadad8896eda9c8c2115970724b15508",
     "grade": true,
     "grade_id": "cell-eb4d5b41bef8f0ab",
     "locked": true,
     "points": 1,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[92m All public tests passed\n"
     ]
    }
   ],
   "source": [
    "utils.test_history(history)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "CubF2J2gSf6q"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "789/789 [==============================] - 0s 22us/sample - loss: 0.3559 - wine_quality_loss: 0.3247 - wine_type_loss: 0.0303 - wine_quality_root_mean_squared_error: 0.5703 - wine_type_accuracy: 0.9949\n",
      "\n",
      "loss: 0.35588227101000847\n",
      "wine_quality_loss: 0.32465794682502747\n",
      "wine_type_loss: 0.03027251549065113\n",
      "wine_quality_rmse: 0.5703111290931702\n",
      "wine_type_accuracy: 0.9949302673339844\n"
     ]
    }
   ],
   "source": [
    "# Gather the training metrics\n",
    "loss, wine_quality_loss, wine_type_loss, wine_quality_rmse, wine_type_accuracy = model.evaluate(x=norm_val_X, y=val_Y)\n",
    "\n",
    "print()\n",
    "print(f'loss: {loss}')\n",
    "print(f'wine_quality_loss: {wine_quality_loss}')\n",
    "print(f'wine_type_loss: {wine_type_loss}')\n",
    "print(f'wine_quality_rmse: {wine_quality_rmse}')\n",
    "print(f'wine_type_accuracy: {wine_type_accuracy}')\n",
    "\n",
    "# EXPECTED VALUES\n",
    "# ~ 0.30 - 0.38\n",
    "# ~ 0.30 - 0.38\n",
    "# ~ 0.018 - 0.036\n",
    "# ~ 0.50 - 0.62\n",
    "# ~ 0.97 - 1.0\n",
    "\n",
    "# Example:\n",
    "#0.3657050132751465\n",
    "#0.3463745415210724\n",
    "#0.019330406561493874\n",
    "#0.5885359048843384\n",
    "#0.9974651336669922"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "gPtTGAP4usnm"
   },
   "source": [
    "## Analyze the Model Performance\n",
    "\n",
    "Note that the model has two outputs. The output at index 0 is quality and index 1 is wine type\n",
    "\n",
    "So, round the quality predictions to the nearest integer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "tBq9PEeAaW-Y"
   },
   "outputs": [],
   "source": [
    "predictions = model.predict(norm_test_X)\n",
    "quality_pred = predictions[0]\n",
    "type_pred = predictions[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "YLhgTR4xTIxj"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.6428285]\n"
     ]
    }
   ],
   "source": [
    "print(quality_pred[0])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# 5.4 - 6.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "MPi-eYfGTUXi"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.00327314]\n",
      "[0.9999658]\n"
     ]
    }
   ],
   "source": [
    "print(type_pred[0])\n",
    "print(type_pred[944])\n",
    "\n",
    "# EXPECTED OUTPUT\n",
    "# A number close to zero\n",
    "# A number close to or equal to 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Kohk-9C6vt_s"
   },
   "source": [
    "### Plot Utilities\n",
    "\n",
    "We define a few utilities to visualize the model performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "62gEOFUhn6aQ"
   },
   "outputs": [],
   "source": [
    "def plot_metrics(metric_name, title, ylim=5):\n",
    "    plt.title(title)\n",
    "    plt.ylim(0,ylim)\n",
    "    plt.plot(history.history[metric_name],color='blue',label=metric_name)\n",
    "    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "6rfgSx7uz5dj"
   },
   "outputs": [],
   "source": [
    "def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    fig = plt.figure()\n",
    "    ax = fig.add_subplot(111)\n",
    "    cax = ax.matshow(cm)\n",
    "    plt.title('Confusion matrix of the classifier')\n",
    "    fig.colorbar(cax)\n",
    "    ax.set_xticklabels([''] + labels)\n",
    "    ax.set_yticklabels([''] + labels)\n",
    "    plt.xlabel('Predicted')\n",
    "    plt.ylabel('True')\n",
    "    fmt = 'd'\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "          plt.text(j, i, format(cm[i, j], fmt),\n",
    "                  horizontalalignment=\"center\",\n",
    "                  color=\"black\" if cm[i, j] > thresh else \"white\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "dfVLIqi017Vf"
   },
   "outputs": [],
   "source": [
    "def plot_diff(y_true, y_pred, title = '' ):\n",
    "    plt.scatter(y_true, y_pred)\n",
    "    plt.title(title)\n",
    "    plt.xlabel('True Values')\n",
    "    plt.ylabel('Predictions')\n",
    "    plt.axis('equal')\n",
    "    plt.axis('square')\n",
    "    plt.plot([-100, 100], [-100, 100])\n",
    "    return plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "8sd1jdFbwE0I"
   },
   "source": [
    "### Plots for Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "f3MwZ5J1pOfj"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_metrics('wine_quality_root_mean_squared_error', 'RMSE', ylim=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "QIAxEezCppnd"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_metrics('wine_type_loss', 'Wine Type Loss', ylim=0.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "uYV9AOAMwI9p"
   },
   "source": [
    "### Plots for Confusion Matrix\n",
    "\n",
    "Plot the confusion matrices for wine type. You can see that the model performs well for prediction of wine type from the confusion matrix and the loss metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "C3hvTYxIaf3n"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_confusion_matrix(test_Y[1], np.round(type_pred), title='Wine Type', labels = [0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "GW91ym8P2I5y"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "scatter_plot = plot_diff(test_Y[0], quality_pred, title='Type')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "include_colab_link": true,
   "name": "exercise-answer.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
