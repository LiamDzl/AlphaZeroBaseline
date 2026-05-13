from selfplay_functions import alphazero_selfplay

if __name__ == '__main__':
    alphazero_selfplay(
        model_path="alphazero.pt",
        cpu_cores=8,
        game_sets=10000,
        epochs=10,
        nabla=0.00002
    )