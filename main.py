import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from project_3d import load_scene_context, draw_label_on_image

class LabelTool:
    def __init__(self, master):
        self.master = master
        master.title("3D Kitti 라벨링")

        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.right_frame, width=1280, height=720)
        self.canvas.pack()
        # 마우스 휠 이벤트 바인딩 (Windows/macOS, Linux)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows, macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

        # 클릭 카운트 및 캔버스 클릭 이벤트 바인딩
        self.select_count = 0
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-2>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_click)


        self.load_button = tk.Button(self.left_frame, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.h_var = tk.DoubleVar(value=1.5)
        self.w_var = tk.DoubleVar(value=1.8)
        self.l_var = tk.DoubleVar(value=4.0)
        self.X_var = tk.DoubleVar(value=0)
        self.Y_var = tk.DoubleVar(value=0)
        self.Z_var = tk.DoubleVar(value=10)
        self.ry_var = tk.DoubleVar(value=0)
        # 추가: 2D 박스 좌표 변수
        self.x1_var = tk.DoubleVar(value=0)
        self.y1_var = tk.DoubleVar(value=0)
        self.x2_var = tk.DoubleVar(value=50)
        self.y2_var = tk.DoubleVar(value=50)
        # truncation, occlusion 변수
        self.truncation_var = tk.DoubleVar(value=0)
        self.occlusion_var = tk.IntVar(value=0)

        # 라벨 리스트 추가
        self.labels = []

        # truncation slider
        frame = tk.Frame(self.left_frame)
        frame.pack()
        tk.Label(frame, text="truncation").pack(side=tk.LEFT)
        scale = tk.Scale(frame, variable=self.truncation_var, orient="horizontal", length=300,
                         from_=0, to=1, resolution=0.1, command=self.update_box)
        scale.pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=self.truncation_var, width=6)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda event: self.update_box(self.truncation_var.get()))

        # occlusion slider
        frame = tk.Frame(self.left_frame)
        frame.pack()
        tk.Label(frame, text="occlusion").pack(side=tk.LEFT)
        scale = tk.Scale(frame, variable=self.occlusion_var, orient="horizontal", length=300,
                         from_=0, to=2, resolution=1, command=self.update_box)
        scale.pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=self.occlusion_var, width=6)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda event: self.update_box(self.occlusion_var.get()))

        for name, var in zip(
            ["x1","y1","x2","y2","h","w","l","X","Y","Z","ry"],
            [self.x1_var,self.y1_var,self.x2_var,self.y2_var,
             self.h_var,self.w_var,self.l_var,self.X_var,self.Y_var,self.Z_var,self.ry_var]):
            frame = tk.Frame(self.left_frame)
            frame.pack()
            tk.Label(frame, text=name).pack(side=tk.LEFT)

            if name == "h":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=1, resolution=0.001, command=self.update_box)
            elif name == "w":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=1, resolution=0.010, command=self.update_box)
            elif name == "l":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=1, resolution=0.001, command=self.update_box)
            elif name == "X" or name == "Y":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=-5, to=5, resolution=0.01, command=self.update_box)
            elif name == "Z":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=5, resolution=0.01, command=self.update_box)
            elif name == "ry":
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=-3.14, to=3.14, resolution=0.01, command=self.update_box)
            elif name in ["x1","y1","x2","y2"]:
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=1920, resolution=1, command=self.update_box)
            elif name in ["y1","y2"]:
                scale = tk.Scale(frame, variable=var, orient="horizontal", length=300,
                                 from_=0, to=1080, resolution=1, command=self.update_box)

            scale.pack(side=tk.LEFT)
            entry = tk.Entry(frame, textvariable=var, width=6)
            entry.pack(side=tk.LEFT)
            entry.bind("<Return>", lambda event, var=var: self.update_box(var.get()))

        # 차량 프리셋 버튼
        preset_frame = tk.Frame(self.left_frame)
        preset_frame.pack()
        tk.Button(preset_frame, text="펠리세이드", command=lambda: self.set_vehicle_preset(0.097, 0.1, 0.265)).pack(side=tk.LEFT)
        tk.Button(preset_frame, text="BMW", command=lambda: self.set_vehicle_preset(0.121, 0.16, 0.353)).pack(side=tk.LEFT)
        tk.Button(preset_frame, text="부가티", command=lambda: self.set_vehicle_preset(0.089, 0.156, 0.324)).pack(side=tk.LEFT)

        # Create a horizontal frame for Add and Delete buttons
        button_frame = tk.Frame(self.left_frame)
        button_frame.pack()

        self.add_button = tk.Button(button_frame, text="Add Object", command=self.add_label)
        self.add_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(button_frame, text="Delete Object", command=self.delete_label)
        self.delete_button.pack(side=tk.LEFT)

        # Save Label 버튼을 아래로
        save_frame = tk.Frame(self.left_frame)
        save_frame.pack()

        self.save_button = tk.Button(save_frame, text="Save Label", command=self.save_label)
        self.save_button.pack()

        # Listbox 추가: Add Object 아래에 라벨 리스트 표시
        self.label_listbox = tk.Listbox(self.left_frame, width=60, height=10)
        self.label_listbox.pack()

        # Listbox에서 선택 시 라벨 로드 기능 바인딩 추가
        self.label_listbox.bind("<<ListboxSelect>>", self.load_label_from_list)

        self.bbox = (0,0,0,0)
        self.mode = 'World'  # 'Ground' or 'World'

        for key in ["<Left>", "<Right>", "<Up>", "<Down>", "a", "d", "w", "s", "W", "A", "S", "D", "q", "e", "Q", "E"]:
            master.bind(key, self.move)

    def load_label_from_list(self, event):
        if not self.label_listbox.curselection():
            return
        index = self.label_listbox.curselection()[0]
        label_line = self.labels[index]
        fields = label_line.split()
        self.truncation_var.set(float(fields[1]))
        self.occlusion_var.set(int(fields[2]))
        self.x1_var.set(int(fields[4]))
        self.y1_var.set(int(fields[5]))
        self.x2_var.set(int(fields[6]))
        self.y2_var.set(int(fields[7]))
        self.h_var.set(float(fields[8]))
        self.w_var.set(float(fields[9]))
        self.l_var.set(float(fields[10]))
        self.X_var.set(float(fields[11]))
        self.Y_var.set(float(fields[12]))
        self.Z_var.set(float(fields[13]))
        self.ry_var.set(float(fields[14]))
        self.update_box(None)

    def set_vehicle_preset(self, h, w, l):
        self.h_var.set(h)
        self.w_var.set(w)
        self.l_var.set(l)
        self.update_box(None)

    def add_label(self):
        label_str = (
            f"Car {self.truncation_var.get():.6f} {self.occlusion_var.get()} 0 "
            f"{int(self.x1_var.get())} {int(self.y1_var.get())} {int(self.x2_var.get())} {int(self.y2_var.get())} "
            f"{self.h_var.get():.4f} {self.w_var.get():.4f} {self.l_var.get():.4f} "
            f"{self.X_var.get():.4f} {self.Y_var.get():.4f} {self.Z_var.get():.4f} {self.ry_var.get():.4f}"
        )
        self.labels.append(label_str)
        self.label_listbox.insert(tk.END, label_str)
        print("Added:", label_str)
        # Entry와 슬라이더를 초기값으로 리셋
        self.truncation_var.set(0)
        self.occlusion_var.set(0)
        self.x1_var.set(0)
        self.y1_var.set(0)
        self.x2_var.set(50)
        self.y2_var.set(50)
        self.h_var.set(1.5)
        self.w_var.set(1.8)
        self.l_var.set(4.0)
        self.X_var.set(0)
        self.Y_var.set(0)
        self.Z_var.set(10)
        self.ry_var.set(0)
        self.update_box(None)

    def load_image(self):
        if hasattr(self, "img_path"):
            keep_labels = messagebox.askyesno("라벨 유지 여부", "라벨링 정보를 유지하시겠습니까?")
            if not keep_labels:
                self.labels = []
                self.label_listbox.delete(0, tk.END)

        path = filedialog.askopenfilename()
        self.img_path = path
        self.name = 'test_img'  # 무조건 'test_img'로 고정

        self.img = cv2.imread(path)
        self.img_orig = self.img.copy()

        # calibration 등은 무조건 main.py 기준의 data_demo 아래에서 불러오기
        self.p2, self.extrinsics, self.denorm = load_scene_context(
            "./data_demo",
            self.name
        )
        self.show_image()

    def show_image(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 비율 유지하며 resize
        w, h = img_pil.size
        max_width, max_height = 1280, 720
        ratio = min(max_width / w, max_height / h, 1.0)  # 1.0 이하만 적용
        new_w, new_h = int(w * ratio), int(h * ratio)
        self.show_scale_ratio = ratio  # 축소 비율 저장
        img_pil = img_pil.resize((new_w, new_h))

        # canvas 크기 맞추기
        self.canvas.config(width=new_w, height=new_h)

        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def update_box(self, val):
        if not hasattr(self, "img_orig"):
            return
        # 항상 원본 해상도에서 라벨링
        self.img = self.img_orig.copy()
        self.img = draw_label_on_image(self.img, self.p2, self.extrinsics, self.denorm,
                                       self.mode,
                                       self.h_var.get(), self.w_var.get(), self.l_var.get(),
                                       self.X_var.get(), self.Y_var.get(), self.Z_var.get(),
                                       self.ry_var.get())
        # 2D bounding box 시각화 추가
        x1, y1, x2, y2 = int(self.x1_var.get()), int(self.y1_var.get()), int(self.x2_var.get()), int(self.y2_var.get())
        cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_image()
 
    def save_label(self):
        img_dir = os.path.dirname(self.img_path)
        label_dir = os.path.join(img_dir, "label_2")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        with open(label_path, "w") as f:
            for label in self.labels:
                f.write(label + "\n")
        print("Saved:", label_path)

    def move(self, event):
        shift_pressed = event.state & 0x0001
        delta = 0.01 if shift_pressed else 0.1

        key = event.keysym.lower()
        char = event.char.lower()

        if key in ["left", "right", "up", "down"]:
            if key == "left":
                self.X_var.set(self.X_var.get() - delta)
            elif key == "right":
                self.X_var.set(self.X_var.get() + delta)
            elif key == "up":
                self.Y_var.set(self.Y_var.get() - delta)
            elif key == "down":
                self.Y_var.set(self.Y_var.get() + delta)
        elif char in ["a", "d", "w", "s"]:
            if char == "a":
                self.X_var.set(self.X_var.get() - delta)
            elif char == "d":
                self.X_var.set(self.X_var.get() + delta)
            elif char == "w":
                self.Y_var.set(self.Y_var.get() - delta)
            elif char == "s":
                self.Y_var.set(self.Y_var.get() + delta)
        elif char in ["q", "e"]:
            if char == "q":
                self.Z_var.set(self.Z_var.get() + delta)
            elif char == "e":
                self.Z_var.set(self.Z_var.get() - delta)
        self.update_box(None)

    def on_canvas_click(self, event):
        if not hasattr(self, "img_orig"):
            return
        shift_pressed = event.state & 0x0001
        if not shift_pressed:
            return  # shift가 눌리지 않으면 아무 동작 안함

        # 클릭 위치를 원본 이미지 비율로 환산
        x = int(event.x / self.show_scale_ratio)
        y = int(event.y / self.show_scale_ratio)

        if event.num == 1:  # 마우스 왼쪽 버튼
            self.x1_var.set(x)
            self.y1_var.set(y)
        elif event.num == 2 or event.num == 3:  # 마우스 오른쪽 버튼
            self.x2_var.set(x)
            self.y2_var.set(y)
        self.update_box(None)

    def on_mouse_wheel(self, event):
        delta = 0
        try:
            if event.num == 4:  # Linux scroll up
                delta = -0.05
            elif event.num == 5:  # Linux scroll down
                delta = 0.05
        except AttributeError:
            pass

        try:
            if event.delta > 0:
                delta = -0.05
            elif event.delta < 0:
                delta = 0.05
        except AttributeError:
            pass

        if delta != 0:
            self.ry_var.set(self.ry_var.get() + delta)
            self.update_box(None)
            
    def delete_label(self):
        if not self.label_listbox.curselection():
            return
        index = self.label_listbox.curselection()[0]
        self.label_listbox.delete(index)
        del self.labels[index]
        print(f"Deleted label at index {index}")

root = tk.Tk()
app = LabelTool(root)
root.mainloop()
