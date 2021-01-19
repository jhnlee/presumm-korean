from others.mecab_rouge import Rouge
import os
import argparse


def format_rouge_scores(scores):
    return """\n
****** ROUGE SCORES ******
** ROUGE 1
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}
** ROUGE 2
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}
** ROUGE L
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}""".format(
        scores["rouge-1"]["f"],
        scores["rouge-1"]["p"],
        scores["rouge-1"]["r"],
        scores["rouge-2"]["f"],
        scores["rouge-2"]["p"],
        scores["rouge-2"]["r"],
        scores["rouge-l"]["f"],
        scores["rouge-l"]["p"],
        scores["rouge-l"]["r"],
    )


def save_rouge_scores(str_scores, path):
    with open(path, "w") as output:
        output.write(str_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_path", type=str, required=True, help="Path for dataset to inference"
    )
    parser.add_argument("--save_path", type=str, default="./results/", help="Path for rouge score")

    args = parser.parse_args()
    candidate_path = args.candidate_path
    gold_path = os.path.splitext(candidate_path)[0] + ".gold"

    candidate = []
    with open(candidate_path, "r") as f:
        for i in f.readlines():
            candidate.append(i.strip())

    gold = []
    with open(gold_path, "r") as f:
        for i in f.readlines():
            gold.append(i.strip())

    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
    )
    scores = rouge_evaluator.get_scores(candidate, gold)
    str_scores = format_rouge_scores(scores)

    save_path = os.path.join(
        args.save_path, os.path.splitext(os.path.split(candidate_path)[-1])[0] + ".rouge"
    )
    print(str_scores)

    save_rouge_scores(str_scores, save_path)
