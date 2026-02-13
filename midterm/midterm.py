import numpy


def inputPair(x):
    match x:
        case 0:
            return (0, 0)
        case 1:
            return (1, 0)
        case 2:
            return (0, 1)
        case 3:
            return (1, 1)


def f(inputPair, weights):
    x = weights[0] * inputPair[0] + weights[1] * inputPair[1] + weights[2]
    return x


def f_or(inputPair):
    if inputPair[0] == 1 or inputPair[1] == 1:
        return 1
    return 0


def gradientDescent(
    inputWeights, iterations, learningRate, activation, activationPrime
):
    weight_buff = inputWeights
    print(inputWeights)
    for j in range(iterations):
        dw_1 = 0
        dw_2 = 0
        db = 0
        for i in range(0, 4):
            z_hat = f(inputPair(i), weight_buff)
            z_hat = activation(z_hat)
            error = z_hat - f_or(inputPair(i))
            grad = error * activationPrime(z_hat)

            dw_1 += grad * inputPair(i)[0]
            dw_2 += grad * inputPair(i)[1]
            db += grad

        weight_buff[0] = weight_buff[0] - learningRate * dw_1
        weight_buff[1] = weight_buff[1] - learningRate * dw_2
        weight_buff[2] = weight_buff[2] - learningRate * db

    return weight_buff


def reluPrime(x):
    if x <= 0:
        return 0
    else:
        return 1


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoidPrime(x):
    s = sigmoid(x)
    return s * (1 + s)


def main():
    learningRate = 0.01
    inputWeights = [1.1, 1.2, 1.4]
    iteration = 100000

    weights = gradientDescent(
        inputWeights, iteration, learningRate, sigmoid, sigmoidPrime
    )
    print(weights)

    for i in range(0, 4):
        print(reluPrime(f(inputPair(i), weights)))


if __name__ == "__main__":
    main()
