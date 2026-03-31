from selfplay_functions import alphazero_selfplay

alphazero_selfplay(model_path="trained.pt", cpu_cores=10, game_sets=10, epochs=10, nabla=0.00002)