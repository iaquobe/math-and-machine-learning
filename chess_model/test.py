import model 
import cairosvg
import chess_util 
from importlib import reload
reload(model)
reload(chess_util)

import torch
import chess
import chess.svg
from model import ChessModel
from chess_util import *

board = chess.Board()
model = ChessModel().to("cpu")


for i in range(40): 
    x    = tensor_from_position(board)
    y    = model(x)
    move = sample_move(board, y)
    board.push(move)

    svg = chess.svg.board(board)
    cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                     write_to="games/game-1-move-{:03d}.png".format(i))
