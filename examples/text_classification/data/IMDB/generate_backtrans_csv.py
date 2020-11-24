def main():
    with open("train_aug.csv", "w", encoding='utf-8') as out_file:
        with open("back_trans.txt", encoding='utf-8') as aug_file:
            with open("train.csv", encoding='utf-8') as train_file:
                headers = ["content", "label", "id"]
                out_file.write("\t".join(headers) + "\n")
                train_file.readline()
                for aug_line in aug_file.readlines():
                    train_line = train_file.readline().strip().split('\t')
                    label = train_line[-2]
                    id = train_line[-1]
                    out_file.write('\t'.join([aug_line.strip(), label, id]) + '\n')


if __name__ == "__main__":
    main()