import numpy as np
import matplotlib.pyplot as plt

class TextAnnotator:
    def __init__(self, config):
        self.config = config
        self.annotations = []
        self.zero_x = np.array([0, 0.1, 0.4, 0.5, 0.5, 0.4, 0.1, 0.05, 0.45, 0.05, 0, 0])
        self.one_x = np.array([0.25, 0.5, 0.5])
        self.two_x = np.array([0, 0.4, 0.5, 0.5, 0.4, 0.1, 0, 0, 0.5])
        self.three_x = np.array([0, 0.4, 0.5, 0.5, 0.4, 0.1, 0.4, 0.5, 0.5, 0.4, 0])
        self.four_x = np.array([0.15, 0, 0.5, 0.4, 0.4, 0.4])
        self.five_x = np.array([0.5, 0, 0, 0.4, 0.5, 0.5, 0.4, 0])
        self.six_x = np.array([0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.4, 0])
        self.seven_x = np.array([0, 0, 0.5, 0.35])
        self.eight_x = np.array([0, 0.1, 0.4, 0.5, 0.5, 0.4, 0.5, 0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.1, 0, 0])
        self.nine_x = np.array([0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.3])

        self.zero_y = np.array([0.9, 1, 1, 0.9, 0.1, 0, 0, 0.05, 0.95, 0.05, 0.1, 0.9])
        self.one_y = np.array([0.75, 1, 0])
        self.two_y = np.array([1, 1, 0.9, 0.6, 0.5, 0.5, 0.4, 0, 0])
        self.three_y = np.array([1, 1, 0.9, 0.6, 0.5, 0.5, 0.5, 0.4, 0.1, 0, 0])
        self.four_y = np.array([1, 0.5, 0.5, 0.5, 1, 0])
        self.five_y = np.array([1, 1, 0.5, 0.5, 0.4, 0.1, 0, 0])
        self.six_y = np.array([0.9, 1, 1, 0.9, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5])
        self.seven_y = np.array([0.75, 1, 1, 0])
        self.eight_y = np.array([0.9, 1, 1, 0.9, 0.6, 0.5, 0.4, 0.1, 0, 0, 0.1, 0.4, 0.5, 0.5, 0.5, 0.6, 0.9])
        self.nine_y = np.array([0.6, 0.6, 0.7, 0.9, 1, 1, 0.9, 0.8, 0])
        self.text_number_x = [self.zero_x, self.one_x, self.two_x, self.three_x, self.four_x, self.five_x, self.six_x,
                              self.seven_x,
                              self.eight_x, self.nine_x]
        self.text_number_y = [self.zero_y, self.one_y, self.two_y, self.three_y, self.four_y, self.five_y, self.six_y,
                              self.seven_y,
                              self.eight_y, self.nine_y]
        self.text_c_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75])
        self.text_c_y = np.array([1, 1, 0.75, 0.25, 0, 0])
        self.text_w_x = np.linspace(0, 1, 5)
        self.text_w_y = np.array([1, 0, 1, 0, 1])
        self.text_i_x = np.array([0.2, 0.8, 0.5, 0.5, 0.8, 0.2])
        self.text_i_y = np.array([0, 0, 0, 1, 1, 1])
        self.text_m_x = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
        self.text_m_y = np.array([0, 1, 0.2, 1, 0])
        self.text_o_x = np.array([0.75, 0.25, 0, 0, 0.25, 0.75, 1, 1, 0.75])
        self.text_o_y = np.array([1, 1, 0.75, 0.25, 0, 0, 0.25, 0.75, 1])
        self.text_e_x = np.array([0.5, 0, 0, 0.5, 0, 0, 0.5])
        self.text_e_y = np.array([1, 1, 0.5, 0.5, 0.5, 0, 0])
        self.text_l_x = np.array([0.25, 0.25, 0.75])
        self.text_l_y = np.array([1.05, 0.3, 0.3])
        self.text_r_x = np.array([0.25, 0.25, 0.65, 0.75, 0.75, 0.65, 0.25, 0.45, 0.75]) * 1.2 - 0.15
        self.text_r_y = np.array([0.4, 1, 1, 0.9, 0.8, 0.7, 0.7, 0.7, 0.4]) * 1.2 - 0.2
        self.text_wing_x = [self.text_c_x, self.text_i_x, self.text_m_x, self.text_o_x, self.text_e_x]
        self.text_wing_y = [self.text_c_y, self.text_i_y, self.text_m_y, self.text_o_y, self.text_e_y]
        self.text_l_r_x = [self.text_l_x, self.text_r_x]
        self.text_l_r_y = [self.text_l_y, self.text_r_y]

    def add_annotation(self, start, end, label):
        """Add an annotation to the text."""
        if start < 0 or end > len(self.text) or start >= end:
            raise ValueError("Invalid start or end indices for annotation.")
        self.annotations.append((start, end, label))

    def get_annotations(self):
        """Return the list of annotations."""
        return self.annotations

    def annotate_text(self):
        """Annotate the text with the added annotations."""
        annotated_text = self.text
        for start, end, label in sorted(self.annotations, key=lambda x: x[0], reverse=True):
            annotated_text = (annotated_text[:start] + f"[{label}]" +
                              annotated_text[start:end] + f"[/{label}]" +
                              annotated_text[end:])
        return annotated_text


    def check_font(self):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
        offset_step = 1.2
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        # 数字
        for i in range(10):
            x = self.text_number_x[i] + i * offset_step
            y = self.text_number_y[i]
            axes[0].plot(x, y, label=str(i), color=colors[i % len(colors)],
                         linewidth=2, marker=markers[i % len(markers)], markersize=6)
        axes[0].set_title('Numbers')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_aspect('equal')

        # アルファベット
        base = 10 * offset_step
        axes[1].plot(self.text_c_x + base, self.text_c_y, label='c', color='black', linewidth=2, marker='o')
        axes[1].plot(self.text_w_x + base + offset_step, self.text_w_y, label='w', color='gray', linewidth=2,
                     marker='s')
        axes[1].plot(self.text_i_x + base + 2 * offset_step, self.text_i_y, label='i', color='pink', linewidth=2,
                     marker='^')
        axes[1].plot(self.text_m_x + base + 3 * offset_step, self.text_m_y, label='m', color='lime', linewidth=2,
                     marker='D')
        axes[1].plot(self.text_o_x + base + 4 * offset_step, self.text_o_y, label='o', color='navy', linewidth=2,
                     marker='v')
        axes[1].plot(self.text_e_x + base + 5 * offset_step, self.text_e_y, label='e', color='teal', linewidth=2,
                     marker='>')
        axes[1].plot(self.text_l_r_x[0] + base + 6 * offset_step, self.text_l_r_y[0], label='l', color='red',
                     linewidth=2, marker='<')
        axes[1].plot(self.text_l_r_x[1] + base + 7 * offset_step, self.text_l_r_y[1], label='r', color='blue',
                     linewidth=2, marker='p')
        axes[1].set_title('Alphabets')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    config = {
        'text': 'WingMaker',
        'font_size': 12,
        'font_color': 'black'
    }
    annotator = TextAnnotator(config)
    annotator.check_font()

