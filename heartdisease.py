{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reading the  heart dataset\n",
    "dataset= pd.read_csv(\"heart.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
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
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#displaying the first 5 rows\n",
    " \n",
    "dataset.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 303 entries, 0 to 302\n",
      "Data columns (total 14 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   age       303 non-null    int64  \n",
      " 1   sex       303 non-null    int64  \n",
      " 2   cp        303 non-null    int64  \n",
      " 3   trestbps  303 non-null    int64  \n",
      " 4   chol      303 non-null    int64  \n",
      " 5   fbs       303 non-null    int64  \n",
      " 6   restecg   303 non-null    int64  \n",
      " 7   thalach   303 non-null    int64  \n",
      " 8   exang     303 non-null    int64  \n",
      " 9   oldpeak   303 non-null    float64\n",
      " 10  slope     303 non-null    int64  \n",
      " 11  ca        303 non-null    int64  \n",
      " 12  thal      303 non-null    int64  \n",
      " 13  target    303 non-null    int64  \n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 33.3 KB\n"
     ]
    }
   ],
   "source": [
    "#cloumns detail information\n",
    "\n",
    "dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    303.000000\n",
       "mean       0.544554\n",
       "std        0.498835\n",
       "min        0.000000\n",
       "25%        0.000000\n",
       "50%        1.000000\n",
       "75%        1.000000\n",
       "max        1.000000\n",
       "Name: target, dtype: float64"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#summary of numerical attributes with repective to target\n",
    "\n",
    "dataset[\"target\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x1080 with 16 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# histograms for each attribute\n",
    "%matplotlib inline  \n",
    "dataset.hist(figsize=(20,15))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "cp          0\n",
       "trestbps    0\n",
       "chol        0\n",
       "fbs         0\n",
       "restecg     0\n",
       "thalach     0\n",
       "exang       0\n",
       "oldpeak     0\n",
       "slope       0\n",
       "ca          0\n",
       "thal        0\n",
       "target      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#check null values in the data\n",
    "\n",
    "dataset.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [],
   "source": [
    "#There are no null values in our data, so the data is clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1    165\n",
      "0    138\n",
      "Name: target, dtype: int64\n",
      "Percentage of patient without heart problems: 45.54\n",
      "Percentage of patient with heart problems: 54.46\n"
     ]
    },
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
    "# count of patients with respect to their heart problem.\n",
    "y = dataset[\"target\"]\n",
    "\n",
    "sns.countplot(y)\n",
    "\n",
    "\n",
    "target_temp = dataset.target.value_counts()\n",
    "\n",
    "print(target_temp)\n",
    "\n",
    "print(\"Percentage of patient without heart problems: \"+str(round(target_temp[0]*100/303,2)))\n",
    "print(\"Percentage of patient with heart problems: \"+str(round(target_temp[1]*100/303,2)))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "target      1.000000\n",
      "exang       0.436757\n",
      "cp          0.433798\n",
      "oldpeak     0.430696\n",
      "thalach     0.421741\n",
      "ca          0.391724\n",
      "slope       0.345877\n",
      "thal        0.344029\n",
      "sex         0.280937\n",
      "age         0.225439\n",
      "trestbps    0.144931\n",
      "restecg     0.137230\n",
      "chol        0.085239\n",
      "fbs         0.028046\n",
      "Name: target, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#correlation of attributes with target\n",
    "print(dataset.corr()[\"target\"].abs().sort_values(ascending=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create dummies for columns of categorical variables.\n",
    "dataset =pd.get_dummies(dataset, columns=['sex','cp','fbs','restecg','exang','exang','slope','ca','thal'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "# transform data and train\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler=StandardScaler()\n",
    "columns_to_scale=['age','trestbps','chol','thalach','oldpeak']\n",
    "dataset[columns_to_scale]=scaler.fit_transform(dataset[columns_to_scale])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
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
       "      <th>age</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>thalach</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>target</th>\n",
       "      <th>sex_0</th>\n",
       "      <th>sex_1</th>\n",
       "      <th>cp_0</th>\n",
       "      <th>cp_1</th>\n",
       "      <th>...</th>\n",
       "      <th>slope_2</th>\n",
       "      <th>ca_0</th>\n",
       "      <th>ca_1</th>\n",
       "      <th>ca_2</th>\n",
       "      <th>ca_3</th>\n",
       "      <th>ca_4</th>\n",
       "      <th>thal_0</th>\n",
       "      <th>thal_1</th>\n",
       "      <th>thal_2</th>\n",
       "      <th>thal_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.952197</td>\n",
       "      <td>0.763956</td>\n",
       "      <td>-0.256334</td>\n",
       "      <td>0.015443</td>\n",
       "      <td>1.087338</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.915313</td>\n",
       "      <td>-0.092738</td>\n",
       "      <td>0.072199</td>\n",
       "      <td>1.633471</td>\n",
       "      <td>2.122573</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.474158</td>\n",
       "      <td>-0.092738</td>\n",
       "      <td>-0.816773</td>\n",
       "      <td>0.977514</td>\n",
       "      <td>0.310912</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.180175</td>\n",
       "      <td>-0.663867</td>\n",
       "      <td>-0.198357</td>\n",
       "      <td>1.239897</td>\n",
       "      <td>-0.206705</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.290464</td>\n",
       "      <td>-0.663867</td>\n",
       "      <td>2.082050</td>\n",
       "      <td>0.583939</td>\n",
       "      <td>-0.379244</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 33 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        age  trestbps      chol   thalach   oldpeak  target  sex_0  sex_1  \\\n",
       "0  0.952197  0.763956 -0.256334  0.015443  1.087338       1      0      1   \n",
       "1 -1.915313 -0.092738  0.072199  1.633471  2.122573       1      0      1   \n",
       "2 -1.474158 -0.092738 -0.816773  0.977514  0.310912       1      1      0   \n",
       "3  0.180175 -0.663867 -0.198357  1.239897 -0.206705       1      0      1   \n",
       "4  0.290464 -0.663867  2.082050  0.583939 -0.379244       1      1      0   \n",
       "\n",
       "   cp_0  cp_1  ...  slope_2  ca_0  ca_1  ca_2  ca_3  ca_4  thal_0  thal_1  \\\n",
       "0     0     0  ...        0     1     0     0     0     0       0       1   \n",
       "1     0     0  ...        0     1     0     0     0     0       0       0   \n",
       "2     0     1  ...        1     1     0     0     0     0       0       0   \n",
       "3     0     1  ...        1     1     0     0     0     0       0       0   \n",
       "4     1     0  ...        1     1     0     0     0     0       0       0   \n",
       "\n",
       "   thal_2  thal_3  \n",
       "0       0       0  \n",
       "1       1       0  \n",
       "2       1       0  \n",
       "3       1       0  \n",
       "4       1       0  \n",
       "\n",
       "[5 rows x 33 columns]"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "predictors = dataset.drop(\"target\",axis=1)\n",
    "target = dataset[\"target\"]\n",
    "\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The accuracy for SvC and Random Forest is: 0.8688524590163934\n"
     ]
    }
   ],
   "source": [
    "#create ensemble model\n",
    "from sklearn.ensemble import RandomForestClassifier, VotingClassifier\n",
    "from sklearn.svm import SVC\n",
    "#from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. \n",
    "svmClf = SVC(probability=True, random_state=0) # probability calculation\n",
    "#logClf = LogisticRegression(random_state=0)\n",
    "#nbclf = GaussianNB(random_state=0)\n",
    "\n",
    "# constructing the ensemble classifier by mentioning the individual classifiers.\n",
    "clf2 = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf)], voting='soft') \n",
    "\n",
    "# train the ensemble classifier\n",
    "clf2.fit(X_train, Y_train)\n",
    "print('The accuracy for SvC and Random Forest is:',clf2.score(X_test,Y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
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
