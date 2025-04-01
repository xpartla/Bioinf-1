import os
from datetime import datetime
import numpy as np
import argparse
from functools import lru_cache
import psutil

def parseArguments():
    parser = argparse.ArgumentParser(description="Better emboss water :) ")

    parser.add_argument("-p", action="store_true", help="Use protein scoring matrix (blossum62.txt)")
    parser.add_argument("-d", action="store_true", help="Use DNA scoring matrix (dna.txt)")
    parser.add_argument("-s", metavar="scoreFile", type=str, help="Specify a custom scoring matrix file")
    parser.add_argument("-r", action="store_true", help="use recursive algorithm instead of smith-waterman")

    parser.add_argument("input_files", nargs=2, help="Two FASTA input files with sequences")

    args = parser.parse_args()

    selectedScoreOptions = [args.p, args.d, args.s is not None]
    if sum(selectedScoreOptions) > 1:
        parser.error("You can use only one -p, -d, or -s <file>.")

    if args.p:
        scoreFile = "blosum62.txt"
    elif args.d:
        scoreFile = "dna.txt"
    elif args.s:
        scoreFile = args.s
    else:
        scoreFile = ''
        parser.error("You must specify a scoring matrix using -p, -d, or -s <file>.")

    firstInput, secondInput = args.input_files

    return firstInput, secondInput, scoreFile, args.r



def readInput(input):
    with open(input, 'r') as file:
        sequence = []
        firstLine = True
        for line in file:
            if firstLine and line.startswith('>'):
                firstLine = False
                sequenceName = line[1:]
                continue
            sequence.append(line.strip())
    return ''.join(sequence), sequenceName

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

def outputScores(seq1, seq2, scores, gapOpen=10, gapExtend=0.5):
    STOP, LEFT, UP, DIAGONAL = 0, 1, 2, 3

    scoreMatrix = np.zeros((len(seq2) + 1, len(seq1) + 1))
    gapX = np.zeros((len(seq2) + 1, len(seq1) + 1))
    gapY = np.zeros((len(seq2) + 1, len(seq1) + 1))
    tracebackMatrix = np.zeros((len(seq2) + 1, len(seq1) + 1), dtype=int)

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            gapX[j, i] = np.maximum(-gapOpen + scoreMatrix[j, i - 1], -gapExtend + gapX[j, i - 1])
            gapY[j, i] = np.maximum(-gapOpen + scoreMatrix[j - 1, i], -gapExtend + gapY[j - 1, i])
            scoreMatrix[j, i] = np.max([
                scores[seq1[i - 1]][seq2[j - 1]] + scoreMatrix[j - 1, i - 1],
                gapX[j, i],
                gapY[j, i],
                0
            ])

            if scoreMatrix[j, i] == 0:
                tracebackMatrix[j, i] = STOP
            elif scoreMatrix[j, i] == gapY[j, i]:
                tracebackMatrix[j, i] = LEFT
            elif scoreMatrix[j, i] == gapX[j, i]:
                tracebackMatrix[j, i] = UP
            elif scoreMatrix[j, i] == scores[seq1[i - 1]][seq2[j - 1]] + scoreMatrix[j - 1, i - 1]:
                tracebackMatrix[j, i] = DIAGONAL

    return gapX, gapY, scoreMatrix, tracebackMatrix


SCORES = {}
SEQ1 = ""
SEQ2 = ""
TRACEBACK = {}
@lru_cache(None)
def recursiveAlignment(i, j):
    if i == 0 or j == 0:
        return 0

    matchScore = SCORES[SEQ1[i - 1]][SEQ2[j - 1]]

    scores = [
        (0, "STOP"),
        (recursiveAlignment(i - 1, j - 1) + matchScore, "DIAG"),
        (recursiveAlignment(i - 1, j) - 10, "UP"),
        (recursiveAlignment(i, j - 1) - 10, "LEFT")
    ]

    bestScore, direction = max(scores, key=lambda x: x[0])
    TRACEBACK[(i, j)] = direction
    return bestScore

def recursiveTraceback(i, j):
    alignedSeq1, alignedSeq2, match = [], [], []

    while (i, j) in TRACEBACK and TRACEBACK[(i, j)] != "STOP":
        if TRACEBACK[(i, j)] == "DIAG":
            alignedSeq1.append(SEQ1[i - 1])
            alignedSeq2.append(SEQ2[j - 1])
            match.append("|" if SEQ1[i - 1] == SEQ2[j - 1] else ".")
            i -= 1
            j -= 1
        elif TRACEBACK[(i, j)] == "UP":
            alignedSeq1.append(SEQ1[i - 1])
            alignedSeq2.append('-')
            match.append(" ")
            i -= 1
        elif TRACEBACK[(i, j)] == "LEFT":
            alignedSeq1.append('-')
            alignedSeq2.append(SEQ2[j - 1])
            match.append(" ")
            j -= 1

    return "".join(reversed(alignedSeq1)), "".join(reversed(match)), "".join(reversed(alignedSeq2))

def traceback(scoreMatrix, tracebackMatrix, seq1, seq2, STOP=0, LEFT=1, UP=2, DIAGONAL=3):
    alignedSeq1, alignedSeq2, match = [], [], []

    maxValue = scoreMatrix.max()
    maxJ, maxI = np.where(scoreMatrix == maxValue)
    maxJ, maxI = int(maxJ[0]), int(maxI[0])

    post1 = seq1[maxI:] if maxI < len(seq1) else ''
    post2 = seq2[maxJ:] if maxJ < len(seq2) else ''

    while tracebackMatrix[maxJ, maxI] != STOP:
        if tracebackMatrix[maxJ, maxI] == DIAGONAL:
            alignedSeq1.append(seq1[maxI - 1])
            alignedSeq2.append(seq2[maxJ - 1])
            match.append("|" if seq1[maxI - 1] == seq2[maxJ - 1] else ".")
            maxI -= 1
            maxJ -= 1
        elif tracebackMatrix[maxJ, maxI] == UP:
            alignedSeq1.append(seq1[maxI - 1])
            alignedSeq2.append('-')
            match.append(" ")
            maxI -= 1
        elif tracebackMatrix[maxJ, maxI] == LEFT:
            alignedSeq1.append('-')
            alignedSeq2.append(seq2[maxJ - 1])
            match.append(" ")
            maxJ -= 1

    pre1 = seq1[:maxI] if maxI > 0 else ''
    pre2 = seq2[:maxJ] if maxJ > 0 else ''

    if len(pre1) < len(pre2):
        pre1 = ' ' * (len(pre2) - len(pre1)) + pre1
    elif len(pre2) < len(pre1):
        pre2 = ' ' * (len(pre1) - len(pre2)) + pre2

    alignedSeq1 = pre1 + '(' + ''.join(alignedSeq1[::-1]) + ')' + post1
    alignedSeq2 = pre2 + '(' + ''.join(alignedSeq2[::-1]) + ')' + post2
    match = ' ' * len(pre1) + '(' + ''.join(match[::-1]) + ')'

    return alignedSeq1, alignedSeq2, match

def outputFile(alignedSeq1, alignedSeq2, match, bestScore, seq1, seq2, name1, name2, runTime, memoryUsage, recursive=False):
    with open('output.txt', 'w') as f:
        f.write('# Rundate: ' + datetime.now().isoformat(timespec='seconds') + '\n')
        f.write('# Execution time: ' + str(runTime) + '\n')
        f.write('# Memory used: ' + str(memoryUsage) + ' MB' + '\n')
        f.write('# gapOpen 10 \n')
        f.write('# gapExtend 0.5 \n\n')
        f.write('# seq_1 : \n - ' + name1 + ' - ' + seq1 + '\n\n# seq_2: \n - ' + name2 + ' - ' + seq2 + '\n\n')
        f.write('# Alignment Score: ' + bestScore + '\n')
        f.write('# Length: ' + str(len(match) - 2) + '\n\n')
        f.write(alignedSeq1 + '\n' + match + '\n' + alignedSeq2 + '\n')
        f.write('\n#-------------------------------------------------\n')
        f.write('#-------------------------------------------------\n')
        if recursive:
            f.write('Computed with recursion')
        else:
            f.write('Computed with dynamic programming')


def measureMemory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024
                                        )

def run(firstInput, secondInput, scoreFile, recursive):
    global SCORES, SEQ1, SEQ2, TRACEBACK
    SEQ1, name1 = readInput(firstInput)
    SEQ2, name2 = readInput(secondInput)
    SCORES = readMatrix(scoreFile)
    TRACEBACK = {}

    startTime = datetime.now()
    startMem = measureMemory()

    if recursive:
        maxScore = 0
        maxI, maxJ = 0, 0

        for i in range(len(SEQ1) + 1):
            for j in range(len(SEQ2) + 1):
                score = recursiveAlignment(i, j)
                if score > maxScore:
                    maxScore = score
                    maxI, maxJ = i, j

        alignedSeq1, match, alignedSeq2 = recursiveTraceback(maxI, maxJ)
        bestScore = str(maxScore)
    else:
        gapX, gapY, scoreMatrix, tracebackMatrix = outputScores(SEQ1, SEQ2, SCORES)
        alignedSeq1, alignedSeq2, match = traceback(scoreMatrix, tracebackMatrix, SEQ1, SEQ2)
        bestScore = str(np.max(scoreMatrix))

    endTime = datetime.now()
    endMem = measureMemory()
    runTime = endTime - startTime
    memoryUsage = endMem - startMem

    outputFile(alignedSeq1, alignedSeq2, match, bestScore, SEQ1, SEQ2, name1, name2, runTime, memoryUsage, recursive)

firstInput, secondInput, scoreFile, recursive = parseArguments()
run(firstInput, secondInput, scoreFile, recursive)