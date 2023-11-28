import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="code to run the sarcasm detection and sarcasm translation model"
    )
    parser.add_argument("-x1", "--x_train", required=True, help="file path to x_training")
    parser.add_argument("-y1", "--y_train", required=True, help="file path to y_training")
    parser.add_argument("-x2", "--x_test", required=True, help="file path to x_test")
    parser.add_argument("-y2", "--y_test", required=False, help="file path to y_test")
    parser.add_argument("-t1", "--train", required=False, help="file path to training set including x and y")
    parser.add_argument("-t2", "--test", required=False, help="file path to test set including x and y")

    args = parser.parse_args()