from sklearn.linear_model import LogisticRegression


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return LogisticRegression(solver='liblinear', random_state=0).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
