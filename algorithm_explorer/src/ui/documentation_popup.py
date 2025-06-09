from tkinter import Toplevel, Label, Button, Text, Scrollbar, END

class DocumentationPopup:
    def __init__(self, master, algorithm_name, algorithm_description):
        self.top = Toplevel(master)
        self.top.title(f"{algorithm_name} Documentation")
        self.top.geometry("400x300")
        self.top.configure(bg="#23272e")

        self.label = Label(self.top, text=algorithm_name, bg="#23272e", fg="#f5f6fa", font=("Segoe UI", 14))
        self.label.pack(pady=10)

        self.text_area = Text(self.top, bg="#2d313a", fg="#f5f6fa", font=("Consolas", 10), wrap="word")
        self.text_area.insert(END, algorithm_description)
        self.text_area.config(state="disabled")
        self.text_area.pack(expand=True, fill="both", padx=10, pady=10)

        self.scrollbar = Scrollbar(self.top, command=self.text_area.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.text_area.config(yscrollcommand=self.scrollbar.set)

        self.close_button = Button(self.top, text="Close", command=self.top.destroy, bg="#353b48", fg="#f5f6fa", font=("Segoe UI", 12))
        self.close_button.pack(pady=10)