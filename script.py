from datetime import datetime
import numpy as np
import pandas as pd
import sys


def parse_arguments():
    if len(sys.argv) < 7 or len(sys.argv) > 8:
        print("Usage: script.py -i1 <input_file_1> -2 <input_file_2> -s <score_file> (-p {protein} | -d {DNA} )")
        sys.exit(1)

    args = {}
    optional_flag = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-i1':
            args['firstInput'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '-i2':
            args['secondInput'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '-s':
            args['score'] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] in ('-p', '-d'):
            if optional_flag is not None:
                print("Error: Only one of -p or -d can be specified.")
                sys.exit(1)
            optional_flag = sys.argv[i]
            i += 1
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)

    if 'firstInput' not in args or 'secondInput' not in args or 'score' not in args:
        print("Error: Missing required arguments -i and -s")
        sys.exit(1)

    if optional_flag is None:
        print("Error: One of -p or -d must be specified")
        sys.exit(1)

    return args['firstInput'], args['secondInput'], args['score'], optional_flag


def readInput(input):
    with open(input, 'r') as file:
        sequence = []
        firstLine = True
        for line in file:
            if firstLine and line.startswith('>'):
                firstLine = False
                continue
            sequence.append(line.strip())
    return ''.join(sequence)


def readMatrix(score):
    with open(score) as f:
        lines = f.readlines()
    scores = {}
    cols = lines[0].split()
    for line in lines[1:len(lines) - 1]:
        items = line.split()
        row = items[0][0]
        row_scores = []
        for item in items[1:]:
            row_scores.append(int(item))
        scores[row] = dict(zip(cols, row_scores))
    return scores


def outputScores(seq1, seq2, scores, gap_open=10, gap_extend=0.5):
    STOP, LEFT, UP, DIAGONAL = 0, 1, 2, 3

    score_matrix = pd.DataFrame(0.0, index=range(len(seq2) + 1), columns=range(len(seq1) + 1))
    gap_x = pd.DataFrame(0.0, index=range(len(seq2) + 1), columns=range(len(seq1) + 1))
    gap_y = pd.DataFrame(0.0, index=range(len(seq2) + 1), columns=range(len(seq1) + 1))
    traceback_matrix = pd.DataFrame(0.0, index=range(len(seq2) + 1), columns=range(len(seq1) + 1))

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            gap_x.iloc[j, i] = max(-gap_open + score_matrix.iloc[j, i - 1], -gap_extend + gap_x.iloc[j, i - 1])
            gap_y.iloc[j, i] = max(-gap_open + score_matrix.iloc[j - 1, i], gap_extend + gap_y.iloc[j - 1, i])
            score_matrix.iloc[j, i] = max(scores[seq1[i - 1]][seq2[j - 1]] + score_matrix.iloc[j - 1, i - 1],
                                          gap_x.iloc[j, i], gap_y.iloc[j, i], 0)

            if score_matrix.iloc[j, i] == 0:
                traceback_matrix.iloc[j, i] = STOP
            elif score_matrix.iloc[j, i] == gap_y.iloc[j, i]:
                traceback_matrix.iloc[j, i] = LEFT
            elif score_matrix.iloc[j, i] == gap_x.iloc[j, i]:
                traceback_matrix.iloc[j, i] = UP
            elif score_matrix.iloc[j, i] == scores[seq1[i - 1]][seq2[j - 1]] + score_matrix.iloc[j - 1, i - 1]:
                traceback_matrix.iloc[j, i] = DIAGONAL

    return gap_x, gap_y, score_matrix, traceback_matrix


def traceback(score_matrix, traceback_matrix, seq1, seq2, STOP=0, LEFT=1, UP=2, DIAGONAL=3):
    aligned_seq1, aligned_seq2, match = [], [], []

    max_value = score_matrix.to_numpy().max()
    max_j, max_i = np.where(score_matrix.to_numpy() == max_value)
    max_j, max_i = int(max_j[0]), int(max_i[0])

    post1 = seq1[max_i:] if max_i < len(seq1) else ''
    post2 = seq2[max_j:] if max_j < len(seq2) else ''

    while traceback_matrix.iloc[max_j, max_i] != STOP:
        if traceback_matrix.iloc[max_j, max_i] == DIAGONAL:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append(seq2[max_j - 1])
            match.append("|" if seq1[max_i - 1] == seq2[max_j - 1] else ".")
            max_i -= 1
            max_j -= 1
        elif traceback_matrix.iloc[max_j, max_i] == UP:
            aligned_seq1.append(seq1[max_i - 1])
            aligned_seq2.append('-')
            match.append(" ")
            max_i -= 1
        elif traceback_matrix.iloc[max_j, max_i] == LEFT:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[max_j - 1])
            match.append(" ")
            max_j -= 1

    pre1 = seq1[:max_i] if max_i > 0 else ''
    pre2 = seq2[:max_j] if max_j > 0 else ''

    if len(pre1) < len(pre2):
        pre1 = ' ' * (len(pre2) - len(pre1)) + pre1
    elif len(pre2) < len(pre1):
        pre2 = ' ' * (len(pre1) - len(pre2)) + pre2

    aligned_seq1 = pre1 + '(' + ''.join(aligned_seq1[::-1]) + ')' + post1
    aligned_seq2 = pre2 + '(' + ''.join(aligned_seq2[::-1]) + ')' + post2
    match = ' ' * len(pre1) + '(' + ''.join(match[::-1]) + ')'

    return aligned_seq1, aligned_seq2, match


def outputFile(aligned_seq1, aligned_seq2, match, score_matrix, seq1, seq2):
    with open('output.txt', 'w') as f:
        f.write('#' * 50 + '\n')
        f.write('Rundate: ' + datetime.now().isoformat(timespec='seconds') + '\n')
        f.write('# seq_1 :' + seq1 + '\n# seq_2: ' + seq2 + '\n')
        f.write('# gapOpen 10 \n')
        f.write('# gapExtend 0.5 \n')
        f.write('#' * 50 + '\n\n')

        f.write('#=================================================\n')
        f.write('# Alignment Score: ' + str(max(score_matrix.max())) + '\n')
        f.write('# Length: ' + str(len(match) - 2) + '\n')
        f.write('#=================================================\n\n')
        f.write(aligned_seq1 + '\n' + match + '\n' + aligned_seq2 + '\n')
        f.write('\n#-------------------------------------------------\n')
        f.write('#-------------------------------------------------\n')


def run(firstInput, secondInput, scoreFile, mode):
    if mode not in ('-p', '-d'):
        return

    seq1 = readInput(firstInput)
    seq2 = readInput(secondInput)

    scores = readMatrix(scoreFile)

    gap_x, gap_y, score_matrix, traceback_matrix = outputScores(seq1, seq2, scores)

    aligned_seq1, aligned_seq2, match = traceback(score_matrix, traceback_matrix, seq1, seq2)

    outputFile(aligned_seq1, aligned_seq2, match, score_matrix, seq1, seq2)


firstInput, secondInput, scoreFile, mode = parse_arguments()
run(firstInput, secondInput, scoreFile, mode)
