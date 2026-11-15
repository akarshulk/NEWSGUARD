
def main():
    test_text = (
        "The government has completely failed the people "
        "with its disastrous and irresponsible policies."
    )

    print("Testing Dbias...")
    print("Input:", test_text)

    try:
        from Dbias.bias_classification import classifier
        from Dbias.bias_recognition import recognizer

        print("\n1. BIAS CLASSIFICATION")
        classification = classifier(test_text)
        print(classification)

        print("\n2. BIAS RECOGNITION")
        recognition = recognizer(test_text)
        print(recognition)

    except Exception as error:
        print("\nDbias test failed:")
        print(type(error).__name__, error)


if __name__ == "__main__":
    main()
