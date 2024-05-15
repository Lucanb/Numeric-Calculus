import os
import subprocess
import tkinter as tk
from tkinter import messagebox, simpledialog

directory_structure = {
    "interface.py": ("interface.py", []),
    "Lab1": {
        "Tema1": {
            "ex1.py": ("Lab1/Tema1/ex1.py", []),
            "ex2.py": ("Lab1/Tema1/ex2.py", []),
            "ex3.py": ("Lab1/Tema1/ex3.py", [])
        }
    },
    "Lab2": {
        "Tema2": {
            "Bonus.py": ("Lab2/Tema2/Bonus.py", [
                ("n", "Enter the dimension of the matrix: "),
                ("t", "Enter the precision level for calculations: ")
            ]),
            "homework.py": ("Lab2/Tema2/homework.py", [
                ("n", "Enter the dimension of the matrix: "),
                ("t", "Enter the precision level for calculations: ")
            ])
        }
    },
    "Lab3": {
        "Tema3": {
            "ex.py": ("Lab3/Tema3/ex.py", [
                ("n", "dimensiunea matricilor sistem : "),
                ("t", "precizia calculelor : ")
            ])
        }
    },
    "Lab4": {
        "Tema4": {
            "new.py": ("Lab4/Tema4/new.py", [])
        }
    },
    "Lab5": {
        "Tema5": {
            "bonus.py": ("Lab5/Tema5/bonus.py", [
                ("n", "Enter the matrix dim: "),
                ("t", "Enter the precision level for calculations: ")
            ]),
            "ex.py": ("Lab5/Tema5/ex.py", [])
        }
    },
    "Lab6": {
        "Tema6": {
            "homework.py": ("Lab6/Tema6/homework.py", [])
        }
    },
    "Lab7": {
        "Tema7": {
            "homework.py": ("Lab7/Tema7/homework.py", [])
        }
    },
    "Lab8": {
        "Tema8": {
            "homework.py": ("Lab8/Tema8/homework.py", [])
        }
    },
    "script.py": ("script.py", [])
}

class ScriptRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Script Runner")
        self.root.geometry("600x800")
        self.root.configure(bg='#f0f0f0')

        frame = tk.Frame(root, bg='#f0f0f0')
        frame.pack(padx=10, pady=10, fill='both', expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')

        canvas = tk.Canvas(frame, bg='#f0f0f0', yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=canvas.yview)

        self.inner_frame = tk.Frame(canvas, bg='#f0f0f0')
        canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')

        self.output_text = tk.Text(root, wrap='word', height=10, bg='#e6e6e6')
        self.output_text.pack(padx=10, pady=10, fill='both', expand=True)

        self.create_buttons(self.inner_frame, directory_structure)

        self.inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

    def create_buttons(self, parent, structure):
        for key, value in structure.items():
            if isinstance(value, dict):
                label = tk.Label(parent, text=key, font=('Arial', 12, 'bold'))
                label.pack(anchor='w', padx=20, pady=5)
                frame = tk.Frame(parent)
                frame.pack(anchor='w', padx=40)
                self.create_buttons(frame, value)
            else:
                script_path, inputs = value
                button = tk.Button(parent, text=key, command=lambda sp=script_path, ip=inputs: self.run_script(sp, ip))
                button.pack(anchor='w', padx=20, pady=2, fill='x')

    def run_script(self, script_path, inputs):
        try:
            input_values = []
            for var, prompt in inputs:
                value = simpledialog.askstring("Input Required", prompt)
                input_values.append(value)
            input_str = "\n".join(input_values)

            process = subprocess.Popen(['python', script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate(input=input_str)

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output)
            if error:
                self.output_text.insert(tk.END, "\nErrors:\n" + error)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run script: {e}")

def main():
    root = tk.Tk()
    app = ScriptRunnerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
