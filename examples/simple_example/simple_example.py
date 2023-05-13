from gmdh import Combi, split_data

X = [[1, 2], [3, 2], [7, 0], [5, 5], [1, 4], [2, 6]]
y = [3, 5, 7, 10, 5, 8]

x_train, x_test, y_train, y_test = split_data(X, y)

# print result arrays
print('x_train:\n', x_train)
print('x_test:\n', x_test)
print('\ny_train:\n', y_train)
print('y_test:\n', y_test)

model = Combi()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

# compare predicted and real value
print()
print('y_predicted: ', y_predicted)
print('y_test: ', y_test)

print()
print('polynomial: ', model.get_best_polynomial())