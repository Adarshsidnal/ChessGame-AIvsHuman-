import pygame
import sys
from const import *
from game import Game
from square import Square
from move import Move
from tensorflow.keras.models import load_model
import numpy as np

# Load AI models for black's moves
from_model = load_model('models/1200-elo/from.h5', compile=False)
to_model = load_model('models/1200-elo/to.h5', compile=False)

class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess')
        self.game = Game()
        self.is_ai_turn = False

    def ai_move(self):
        # Get board state
        board = self.game.board

        # Generate possible moves for AI
        moves = []
        for row in range(8):
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece() and square.piece.color == 'black':
                    piece = square.piece
                    piece_moves = board.calc_moves(piece, row, col, bool=False)
                    moves.extend(piece_moves)
        
        if not moves:
            return  # No moves available

        # Select a move based on AI models (simplified for illustration)
        # Convert moves into input features for the models
        from_positions = np.array([[move.initial.row, move.initial.col] for move in moves])
        to_positions = np.array([[move.final.row, move.final.col] for move in moves])
        
        # Use models to predict the best move
        from_scores = from_model.predict(from_positions)
        to_scores = to_model.predict(to_positions)
        
        # Select the best move based on model predictions
        best_move_index = np.argmax(from_scores + to_scores)
        best_move = moves[best_move_index]
        
        # Make the move on the board
        board.move(best_move.piece, best_move)
        self.game.show_bg(self.screen)
        self.game.show_last_move(self.screen)
        self.game.show_pieces(self.screen)
        self.game.next_turn()

    def mainloop(self):
        screen = self.screen
        game = self.game
        board = self.game.board
        dragger = self.game.dragger

        while True:
            # show methods
            game.show_bg(screen)
            game.show_last_move(screen)
            game.show_moves(screen)
            game.show_pieces(screen)
            game.show_hover(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            # AI move
            if game.next_player == 'black':
                self.ai_move()
                self.is_ai_turn = False  # End AI turn

            for event in pygame.event.get():
                if game.next_player == 'white':
                    # Human move handling
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        dragger.update_mouse(event.pos)
                        clicked_row = dragger.mouseY // SQSIZE
                        clicked_col = dragger.mouseX // SQSIZE

                        if board.squares[clicked_row][clicked_col].has_piece():
                            piece = board.squares[clicked_row][clicked_col].piece
                            if piece.color == game.next_player:
                                board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                                dragger.save_initial(event.pos)
                                dragger.drag_piece(piece)
                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_moves(screen)
                                game.show_pieces(screen)

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if dragger.dragging:
                            dragger.update_mouse(event.pos)
                            released_row = dragger.mouseY // SQSIZE
                            released_col = dragger.mouseX // SQSIZE
                            initial = Square(dragger.initial_row, dragger.initial_col)
                            final = Square(released_row, released_col)
                            move = Move(initial, final)

                            if board.valid_move(dragger.piece, move):
                                captured = board.squares[released_row][released_col].has_piece()
                                board.move(dragger.piece, move)
                                board.set_true_en_passant(dragger.piece)
                                game.play_sound(captured)
                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_pieces(screen)
                                game.next_turn()

                        dragger.undrag_piece()

                # Key press events
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        game.change_theme()
                    elif event.key == pygame.K_r:
                        game.reset()
                        self.__init__()  # Reinitialize the game

                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()

main = Main()
main.mainloop()
