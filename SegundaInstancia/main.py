import numpy as np
import random
from collections import defaultdict
import os


GRID_HEIGHT = 10
GRID_WIDTH = 6
PIECE_LIMIT = 10 

PIECES = {
    'SQUARE': [[(0, 0), (0, 1), (1, 0), (1, 1)]],  # 2x2
    'L': [
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
    ],
    'BAR': [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ],
    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
}
PIECE_IDS = {name: i for i, name in enumerate(PIECES.keys())}
ID_TO_PIECE_NAME = {i: name for name, i in PIECE_IDS.items()}


class TetrisEnv:
    def __init__(self, pieces_per_game=5):
        self.pieces_per_game = pieces_per_game
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.piece_counts = {name: self.pieces_per_game for name in PIECES}
        self.game_over = False

    def reset(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        self.piece_counts = {name: self.pieces_per_game for name in PIECES}
        self.game_over = False
        return self._get_state()

    def _get_state(self):
        skyline = [GRID_HEIGHT] * GRID_WIDTH
        for col in range(GRID_WIDTH):
            for row in range(GRID_HEIGHT):
                if self.grid[row, col] != 0:
                    skyline[col] = row
                    break
        return tuple(skyline)

    def _is_valid_position(self, piece_shape, row_offset, col_offset):
        for r, c in piece_shape:
            abs_r, abs_c = row_offset + r, col_offset + c
            if not (0 <= abs_r < GRID_HEIGHT and 0 <= abs_c < GRID_WIDTH and self.grid[abs_r, abs_c] == 0):
                return False
        return True
    
    def _clear_lines(self):
        lines_cleared = 0
        new_grid = np.zeros_like(self.grid)
        new_row_index = GRID_HEIGHT - 1
        for row in range(GRID_HEIGHT - 1, -1, -1):
            if not np.all(self.grid[row] != 0): # Si la línea no está completa
                new_grid[new_row_index] = self.grid[row]
                new_row_index -= 1
            else:
                lines_cleared += 1
        self.grid = new_grid
        return lines_cleared

    def get_possible_actions(self):
        possible_actions = []
        for piece_name, count in self.piece_counts.items():
            if count > 0:
                piece_id = PIECE_IDS[piece_name]
                for rot_idx, shape in enumerate(PIECES[piece_name]):
                    max_col = GRID_WIDTH - max(c for r, c in shape) #hasta que columna puedo desplazarla sin que se me salga :D 
                    for col in range(max_col):
                         # Una acción es válida si la pieza puede ser colocada
                         possible_actions.append((piece_id, rot_idx, col))
        return possible_actions

    def _count_holes(self):
        holes = 0
        for col in range(GRID_WIDTH):
            found_block = False
            for row in range(GRID_HEIGHT):
                if self.grid[row, col] != 0:
                    found_block = True
                elif found_block and self.grid[row, col] == 0:
                    holes += 1
        return holes

    def _get_skyline_properties(self):
        skyline = self._get_state()
        heights = [GRID_HEIGHT - h for h in skyline]
        total_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights) - 1))
        return total_height, bumpiness

    def step(self, action):
        if self.game_over:
            return self._get_state(), 0, True

        piece_id, rotation_idx, col_offset = action
        piece_name = ID_TO_PIECE_NAME[piece_id]
        
        if self.piece_counts[piece_name] == 0:
            return self._get_state(), -10, True

        shape = PIECES[piece_name][rotation_idx]

        final_row = -1
        for row in range(-min(r for r,c in shape), GRID_HEIGHT):
            if self._is_valid_position(shape, row, col_offset):
                final_row = row
            else:
                break
        
        if final_row == -1:
            self.game_over = True
            return self._get_state(), -100, True

        # Colocar la pieza
        for r, c in shape:
            self.grid[final_row + r, col_offset + c] = piece_id + 1
        
        self.piece_counts[piece_name] -= 1

        lines_cleared = self._clear_lines()
        
        # Calculo de la Recompensa
        reward = 5  # Recompensa base por colocar una pieza
        
        # 1. Recompensa por limpiar líneas
        reward += lines_cleared ** 2 * 10

        # 2. Penalizaciones para guiar al agente
        holes = self._count_holes()
        total_height, bumpiness = self._get_skyline_properties()

        reward -= holes * 0.5          # Penalización por crear huecos
        reward -= total_height * 0.1   # Penalización por la altura total
        reward -= bumpiness * 0.2      # Penalización por la rugosidad del skyline

        done = all(count == 0 for count in self.piece_counts.values()) or self.game_over
        
        return self._get_state(), reward, done

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Tablero:")
        print("+" + "-" * GRID_WIDTH + "+")
        for r in range(GRID_HEIGHT):
            print("|" + "".join([str(cell) if cell != 0 else " " for cell in self.grid[r]]) + "|")
        print("+" + "-" * GRID_WIDTH + "+")
        print("Piezas restantes:", self.piece_counts)


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        # Q-Table: estado -> {acción: valor}
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha       # Tasa de aprendizaje
        self.gamma = gamma       # Factor de descuento
        self.epsilon = epsilon     # Tasa de exploración
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state, possible_actions):
        if not possible_actions:
            return None

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)  # Explorar
        else:
            # Explotar
            q_values = {action: self.q_table[state][action] for action in possible_actions}
            return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, next_possible_actions):
        old_value = self.q_table[state][action]
        
        future_max_q = 0
        if next_possible_actions:
             future_max_q = max(self.q_table[next_state][next_action] for next_action in next_possible_actions)

        new_value = old_value + self.alpha * (reward + self.gamma * future_max_q - old_value)
        self.q_table[state][action] = new_value

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

import time

def main():
    env = TetrisEnv(pieces_per_game=4)
    agent = QLearningAgent()
    num_episodes = 5000

    print(" Iniciando Entrenamiento ...")
    print(f"Episodios: {num_episodes}, Piezas por episodio: {env.pieces_per_game * len(PIECES)}")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            possible_actions = env.get_possible_actions()
            action = agent.choose_action(state, possible_actions)
            
            if action is None:
                break

            next_state, reward, done = env.step(action)
            total_reward += reward

            next_possible_actions = env.get_possible_actions()
            agent.learn(state, action, reward, next_state, next_possible_actions)
            
            state = next_state
        
        agent.update_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episodio: {episode + 1}/{num_episodes} | Recompensa: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("\n--- Entrenamiento Completado ---")

    # --- Evaluación ---
    print("Ejecutando una partida final con la política aprendida (sin exploración)...")
    agent.epsilon = 0 # Desactivar exploración
    state = env.reset()
    done = False
    total_pieces = env.pieces_per_game * len(PIECES)
    
    while not done:
        env.render()
        possible_actions = env.get_possible_actions()
        action = agent.choose_action(state, possible_actions)

        if action is None:
            print("No hay más movimientos posibles.")
            break
        
        piece_name = ID_TO_PIECE_NAME[action[0]]
        print(f"Acción elegida: Poner '{piece_name}' en columna {action[2]}")
        time.sleep(0.5)

        state, reward, done = env.step(action)

    print("\n--- Partida Finalizada ---")
    env.render()
    
    espacios_libres = np.sum(env.grid == 0)
    bloques_colocados = GRID_WIDTH*GRID_HEIGHT - espacios_libres
    piezas_usadas = total_pieces - sum(env.piece_counts.values())

    print("\n--- Análisis de Desempeño ---")
    print(f"Bloques colocados: {bloques_colocados}")
    print(f"Piezas usadas: {piezas_usadas} / {total_pieces}")
    print(f"Total de espacios vacíos: {espacios_libres}")
    print(f"Eficiencia de espacio: {(bloques_colocados / (GRID_WIDTH*GRID_HEIGHT))*100:.2f}%")


if __name__ == "__main__":
    main()
