import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from pandas import Dataframe


df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt', header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#print(df.head())
#print(df.describe())
#print("number of rows = ", df.shape[0])
#print("number of cols = ", df.shape[1])

cormat = df.corr()
print(cormat)
hm= pd.DataFrame(df.corr())
plt.pcolor(hm)
plt.title("Correlation Matrix")
plt.xlabel("features")
plt.ylabel("features")
plt.show()


cols = df.columns
sns.pairplot(df[cols], size=2.5)
#plt.tight_layout()

plt.figure()
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=False,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.xlabel("features")
plt.ylabel("features")
plt.show()


y = df['MEDV'].values

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.figure()
plt.scatter(y_train_pred, y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.title("Linear Regression Residual Plot")
plt.show()
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

alpha_space = np.logspace(-4,0,50)
ridge_scores = []
ridge_scores_std = []

ridge = Ridge(normalize=True)
ridge.fit(X_train, y_train)
ridge_y_train_pred = ridge.predict(X_train)
ridge_y_test_pred = ridge.predict(X_test)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
    
# Display the plot
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    std_error = cv_scores_std / np.sqrt(10)
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.title("CV Scores VS Alpha")
    plt.show()
    
display_plot(ridge_scores, ridge_scores_std)

plt.figure()
plt.scatter(ridge_y_train_pred, ridge_y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(ridge_y_test_pred, ridge_y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.title("Ridge Regression Residual Plot")
plt.show()


print('the best alpha is: ',alpha_space[np.argmax(ridge_scores)])
print('Slope: %.3f' % ridge.coef_[0])
print('Intercept: %.3f' % ridge.intercept_)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, ridge_y_train_pred),mean_squared_error(y_test, ridge_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, ridge_y_train_pred),r2_score(y_test, ridge_y_test_pred)))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.1,normalize = False)
lasso.fit(X_train, y_train)
lasso_y_train_pred = lasso.predict(X_train)
lasso_y_test_pred = lasso.predict(X_test)

lasso_scores = []
lasso_scores_std = []
alpha_space = np.logspace(-4,0,50)
for alpha in alpha_space:
    lasso.alpha = alpha
    lasso_cv_scores = cross_val_score(lasso, X, y, cv=10)
    lasso_scores.append(np.mean(lasso_cv_scores))
    lasso_scores_std.append(np.std(lasso_cv_scores))

display_plot(lasso_scores, lasso_scores_std)

plt.figure()
plt.scatter(lasso_y_train_pred, lasso_y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(lasso_y_test_pred, lasso_y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.title("Lasso Regression Residual Plot")
plt.show()

#names = df.drop('MEDV', axis = 1).columns
#lasso = Lasso(alpha = 0.1)
#lasso_coef = lasso.fit(X,y).coef_
#_ = plt.plot(range(len(names)), lasso_coef)
#_ = plt.xticks(range(len(names)),names, rotation = 60)
#_ = plt.ylabel('Coefficient')
#plt.show()

print('the best alpha is: ',alpha_space[np.argmax(lasso_scores)])
print('Slope: %.3f' % lasso.coef_[0])
print('Intercept: %.3f' % lasso.intercept_)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, lasso_y_train_pred),mean_squared_error(y_test, lasso_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, lasso_y_train_pred),r2_score(y_test, lasso_y_test_pred)))


from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet.fit(X_train, y_train)
elanet_y_train_pred = elanet.predict(X_train)
elanet_y_test_pred = elanet.predict(X_test)

elanet_scores = []
elanet_scores_std = []

l1_space = np.logspace(-4,0,50)
for l1_ratio in l1_space:
    elanet.l1_ratio = l1_ratio
    elanet_cv_scores = cross_val_score(elanet, X, y, cv=10)
    elanet_scores.append(np.mean(elanet_cv_scores))
    elanet_scores_std.append(np.std(elanet_cv_scores))

def elanet_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    std_error = cv_scores_std / np.sqrt(10)
    ax.fill_between(l1_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('l1_ratio')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([l1_space[0], l1_space[-1]])
    ax.set_xscale('log')
    plt.title("CV Scores VS l1_ratio")
    plt.show()

elanet_plot(elanet_scores, elanet_scores_std)

plt.figure()
plt.scatter(elanet_y_train_pred, elanet_y_train_pred - y_train,c='steelblue', marker='o', edgecolor='white',label='Training data')
plt.scatter(elanet_y_test_pred, elanet_y_test_pred - y_test,c='limegreen', marker='s', edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.title("ElasticNet Regression Residual Plot")
plt.show()

print('the best l1_ratio is: ',l1_space[np.argmax(elanet_scores)])
print('Slope: %.3f' % elanet.coef_[0])
print('Intercept: %.3f' % elanet.intercept_)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, elanet_y_train_pred),mean_squared_error(y_test, elanet_y_test_pred)))
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, elanet_y_train_pred),r2_score(y_test, elanet_y_test_pred)))

print("My name is {Yue Liu}")
print("My NetID is: {yueliu6}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")













































