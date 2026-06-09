from datasets import load_dataset


def get_label_name(dataset, tag_id):
    label_feature = dataset["train"].features["ner_tags"].feature
    return label_feature.names[tag_id]


def main():
    print("Downloading tner/medmentions from Hugging Face...")
    dataset = load_dataset("tner/medmentions")

    first_item = dataset["train"][0]
    tokens = [str(token).strip() for token in first_item["tokens"]]
    tag_ids = first_item["ner_tags"]

    rows = []
    for token, tag_id in zip(tokens, tag_ids):
        label_name = get_label_name(dataset, tag_id)
        if label_name == "O":
            meaning = "Normal Word"
        elif label_name.startswith("B-"):
            meaning = "Medical Condition Start"
        elif label_name.startswith("I-"):
            meaning = "Medical Condition Continued"
        else:
            meaning = "Entity Tag"

        rows.append((token, label_name, meaning))

    word_width = max(len("Word"), max(len(word) for word, _, _ in rows))
    label_width = max(len("Label"), max(len(label) for _, label, _ in rows))

    print("\nFirst training example from tner/medmentions\n")
    for word, label, meaning in rows:
        print(f"Word: {word:<{word_width}} -> Label: {label:<{label_width}} ({meaning})")


if __name__ == "__main__":
    main()
