import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import subprocess
import os
import csv
import zipfile
import threading

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and CSV Viewer")
        self.root.geometry("600x600")  

        self.image_folder = "Images"
        self.images = []
        self.current_image_index = 0
        self.input_name= "no image"
        self.output_name= "no image"
        self.delete_all_files_in_image_folder()
        self.create_widgets()
        self.refresh_image_folder()
        self.schedule_image_folder_refresh()

    def delete_all_files_in_image_folder(self):
        try:
            for filename in os.listdir(self.image_folder):
                file_path = os.path.join(self.image_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"All files in {self.image_folder} have been deleted.")
        except Exception as e:
            print(f"Failed to delete files in {self.image_folder}: {e}")

    def create_widgets(self):
        self.select_image_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_image_button.place(relx=0, rely=0, relwidth=0.5, relheight=0.05)

        self.input_image_label = tk.Label(self.root, text="Input Image")
        self.input_image_label.place(relx=0, rely=0.05, relwidth=0.5, relheight=0.05)

        self.image_label = tk.Label(self.root, text="No image selected", bg="lightgray")
        self.image_label.place(relx=0, rely=0.10, relwidth=0.5, relheight=0.4)

        self.output_label = tk.Label(self.root, text="Output")
        self.output_label.place(relx=0.5, rely=0.05, relwidth=0.5, relheight=0.05)

        self.image_panel = tk.Label(self.root, bg="lightgray")
        self.image_panel.place(relx=0.5, rely=0.10, relwidth=0.5, relheight=0.4)

        self.input_image_name = tk.Label(self.root, text=self.input_name)
        self.input_image_name.place(relx=0, rely=0.50, relwidth=0.5, relheight=0.05)

        self.open_program_button = tk.Button(self.root, text="Open Program with Image", command=self.open_program)
        self.open_program_button.place(relx=0, rely=0.55, relwidth=0.5, relheight=0.05)

        self.output_image_name = tk.Label(self.root, text=self.output_name)
        self.output_image_name.place(relx=0.5, rely=0.50, relwidth=0.5, relheight=0.05)

        self.save_images_button = tk.Button(self.root, text="Save Images to Zip", command=self.save_images_to_zip)
        self.save_images_button.place(relx=0.5, rely=0.55, relwidth=0.5, relheight=0.05)

        self.table = ttk.Treeview(self.root, columns=('Column1', 'Column2'), show='headings')
        self.table.heading('Column1', text='NAME')
        self.table.heading('Column2', text='VALUE')
        self.table.place(relx=0, rely=0.60, relwidth=1, relheight=0.40)

        # Bind arrow keys to navigate images
        self.root.bind('<Left>', self.show_previous_image)
        self.root.bind('<Right>', self.show_next_image)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if self.image_path:
            self.image_label.config(text=self.image_path)
            self.show_selected_image()
            self.input_image_name.config(text= os.path.basename(self.image_path))

    def show_selected_image(self):
        try:
            img = Image.open(self.image_path)
            img = img.resize((250, 250), Image.LANCZOS)  # Adjust size as needed
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            self.output_name= os.path.basename(self.image_path)
            print(f"Uploaded image: {self.image_path}")  # Print the name of the uploaded image
        except Exception as e:
            print(f"Failed to load image: {e}")

    def open_program(self):
        self.clear_table()
        self.delete_all_files_in_image_folder()
        if hasattr(self, 'image_path'):
            thread = threading.Thread(target=self.run_program)
            thread.start()

    def run_program(self):
        try:
            subprocess.run(['opencv/OpenCVApplication', self.image_path], check=True)
            self.load_csv_data('Images/test_scoresss.csv')  # Change this to your CSV file path
        except subprocess.CalledProcessError as e:
            self.output_image_name.config(text= f"Failed to open program: {e}")
            print(f"Failed to open program: {e}")

    def clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

    def load_csv_data(self, csv_file):
        try:
            with open(csv_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.table.insert('', 'end', values=row)
        except FileNotFoundError:
            print(f"CSV file not found: {csv_file}")
        except Exception as e:
            print(f"Failed to load CSV data: {e}")

    def refresh_image_folder(self):
        try:
            self.images = [f for f in os.listdir(self.image_folder) if
                           f.lower().endswith(('png', 'jpg', 'jpeg', 'gif','bmp'))]
            if not self.images:
                print("No images found in the specified folder.")
                self.image_panel.config(image='', text='No images available')
            else:
                self.show_image()
        except Exception as e:
            print(f"Failed to refresh image folder: {e}")

    def schedule_image_folder_refresh(self):
        self.refresh_image_folder()
        self.root.after(1000, self.schedule_image_folder_refresh)  # Refresh every 5000 milliseconds (5 seconds)

    def show_image(self):
        if self.images:
            image_path = os.path.join(self.image_folder, self.images[self.current_image_index])
            print(f"Showing image: {image_path}")  # Print the name of the image being shown
            try:
                img = Image.open(image_path)
                img = img.resize((250, 250), Image.LANCZOS)  # Adjust size as needed
                img = ImageTk.PhotoImage(img)
                self.image_panel.config(image=img)
                self.image_panel.image = img
                self.output_image_name.config(text= os.path.basename(image_path))
            except Exception as e:
                print(f"Failed to load image: {e}")
        else:
            print("No images to display.")

    def show_previous_image(self, event):
        if self.images:
            self.current_image_index = (self.current_image_index - 1) % len(self.images)
            self.show_image()

    def show_next_image(self, event):
        if self.images:
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.show_image()

    def save_images_to_zip(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
        if save_path:
            with zipfile.ZipFile(save_path, 'w') as zipf:
                if hasattr(self, 'image_path') and self.image_path:
                    zipf.write(self.image_path, os.path.basename(self.image_path))
                for foldername, subfolders, filenames in os.walk(self.image_folder):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        zipf.write(file_path, os.path.relpath(file_path, self.image_folder))
            print(f"Images saved to {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
