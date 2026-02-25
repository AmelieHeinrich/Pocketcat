import math
import os
import subprocess
import threading
import time
import tkinter as tk

import coremltools as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from torch.utils.data import DataLoader, TensorDataset

L = 14
ENCODED_DIM = 2 + 4 * L


def positional_encode(x: torch.Tensor) -> torch.Tensor:
    freqs = [2**i for i in range(L)]
    encoded = [x]
    for f in freqs:
        encoded += [torch.sin(f * math.pi * x), torch.cos(f * math.pi * x)]
    return torch.cat(encoded, dim=-1)


class TwoInputNeuralBRDFModel(nn.Module):
    def __init__(self, hidden=128, depth=3):
        super().__init__()
        layers = [nn.Linear(ENCODED_DIM, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(positional_encode(x))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class TwoInputNeuralBRDFModelTraining:
    def __init__(self, path):
        img = Image.open(path).convert("RGB")
        img_tensor = TF.to_tensor(img)
        dimension = img_tensor.shape[-1]

        u = torch.linspace(-1.0, 1.0, dimension)
        v = torch.linspace(-1.0, 1.0, dimension)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        self.X = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=1)

        r = img_tensor[0].flatten()
        g = img_tensor[1].flatten()
        b = img_tensor[2].flatten()
        self.Y = torch.stack([r, g, b], dim=1)

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def infer_tensor(self, model: TwoInputNeuralBRDFModel):
        with torch.no_grad():
            y_pred = model(self.X)
        dimension = int(self.X.shape[0] ** 0.5)
        r = y_pred[:, 0].reshape(dimension, dimension)
        g = y_pred[:, 1].reshape(dimension, dimension)
        b = y_pred[:, 2].reshape(dimension, dimension)
        img_tensor = torch.stack([r, g, b], dim=0)
        return TF.to_pil_image(img_tensor.clamp(0, 1))

    def infer(self, model: TwoInputNeuralBRDFModel, path: str):
        img = self.infer_tensor(model)
        img.save(path)

    def loss(self, pred, target):
        return self.mse(pred, target) + 0.1 * self.l1(pred, target)

    def train(
        self,
        model: TwoInputNeuralBRDFModel,
        lr: float,
        epochs: int,
        batch_size: int = 4096,
        epoch_callback=None,
    ):
        dataset = TensorDataset(self.X, self.Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = self.loss
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, Y_batch in loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            if epoch_callback:
                epoch_callback(epoch, avg_loss)
        return losses


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neural BRDF Trainer")
        self.resizable(True, True)

        self.trainer = None
        self.model = None
        self.losses = []
        self.training = False

        self._build_ui()

    def _build_ui(self):
        # Left panel: controls
        ctrl = tk.Frame(self, padx=12, pady=12)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl, text="Neural BRDF Trainer", font=("Helvetica", 14, "bold")).pack(
            anchor="w", pady=(0, 12)
        )

        tk.Label(ctrl, text="Texture path:").pack(anchor="w")
        self.texture_path_var = tk.StringVar(value="Assets/TestTexture.png")
        tk.Entry(ctrl, textvariable=self.texture_path_var, width=30).pack(
            anchor="w", pady=(0, 8)
        )

        tk.Label(ctrl, text="Export name:").pack(anchor="w")
        self.export_name_var = tk.StringVar(value="2_input_neural_brdf")
        tk.Entry(ctrl, textvariable=self.export_name_var, width=30).pack(
            anchor="w", pady=(0, 8)
        )

        tk.Label(ctrl, text="Layer size (hidden):").pack(anchor="w")
        self.hidden_var = tk.IntVar(value=128)
        tk.Spinbox(
            ctrl,
            from_=16,
            to=1024,
            increment=16,
            textvariable=self.hidden_var,
            width=10,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(ctrl, text="Depth:").pack(anchor="w")
        self.depth_var = tk.IntVar(value=3)
        tk.Spinbox(ctrl, from_=1, to=12, textvariable=self.depth_var, width=10).pack(
            anchor="w", pady=(0, 8)
        )

        tk.Label(ctrl, text="Epochs:").pack(anchor="w")
        self.epochs_var = tk.IntVar(value=100)
        tk.Spinbox(
            ctrl,
            from_=1,
            to=10000,
            increment=50,
            textvariable=self.epochs_var,
            width=10,
        ).pack(anchor="w", pady=(0, 8))

        tk.Label(ctrl, text="Learning rate:").pack(anchor="w")
        self.lr_var = tk.DoubleVar(value=0.01)
        tk.Entry(ctrl, textvariable=self.lr_var, width=10).pack(
            anchor="w", pady=(0, 16)
        )

        self.bake_btn = tk.Button(
            ctrl,
            text="Bake",
            command=self._start_training,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=8,
            pady=4,
        )
        self.bake_btn.pack(anchor="w")

        self.status_label = tk.Label(ctrl, text="", wraplength=200, justify="left")
        self.status_label.pack(anchor="w", pady=(8, 0))

        self.progress_label = tk.Label(ctrl, text="", fg="gray")
        self.progress_label.pack(anchor="w")

        # Stats panel
        tk.Frame(ctrl, height=1, bg="#ccc").pack(fill=tk.X, pady=(12, 8))
        tk.Label(ctrl, text="Stats", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.stat_train_time = self._stat_row(ctrl, "Train time")
        self.stat_epoch_speed = self._stat_row(ctrl, "Epoch speed")
        self.stat_infer_time = self._stat_row(ctrl, "Inference time")

        # Right panel: results
        results = tk.Frame(self, padx=8, pady=12)
        results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Images row
        img_row = tk.Frame(results)
        img_row.pack(fill=tk.X)

        self.gt_frame = self._image_panel(img_row, "Ground Truth")
        self.gt_frame.pack(side=tk.LEFT, padx=8)

        self.inferred_frame = self._image_panel(img_row, "Inferred")
        self.inferred_frame.pack(side=tk.LEFT, padx=8)

        # Loss graph
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=results)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(12, 0))

    def _stat_row(self, parent, label):
        row = tk.Frame(parent)
        row.pack(anchor="w", fill=tk.X)
        tk.Label(row, text=f"{label}:", fg="gray", width=14, anchor="w").pack(
            side=tk.LEFT
        )
        val = tk.Label(row, text="—", anchor="w")
        val.pack(side=tk.LEFT)
        return val

    def _image_panel(self, parent, label):
        frame = tk.Frame(parent)
        tk.Label(frame, text=label, font=("Helvetica", 11, "bold")).pack()
        lbl = tk.Label(frame, bg="#222", width=256, height=256)
        lbl.pack()
        frame._img_label = lbl
        return frame

    def _set_image(self, panel, pil_img):
        pil_img = pil_img.resize((256, 256), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(pil_img)
        panel._img_label.configure(image=tk_img, width=256, height=256)
        panel._img_label._tk_img = tk_img  # prevent GC

    def _start_training(self):
        if self.training:
            return
        texture_path = self.texture_path_var.get()
        if not os.path.exists(texture_path):
            self.status_label.config(
                text=f"Texture not found: {texture_path}", fg="red"
            )
            return

        self.training = True
        self.bake_btn.config(state=tk.DISABLED)
        self.losses = []
        self.status_label.config(text="Training...", fg="black")
        self.progress_label.config(text="")
        self.stat_train_time.config(text="—")
        self.stat_epoch_speed.config(text="—")
        self.stat_infer_time.config(text="—")
        self.ax.clear()
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.canvas.draw()

        # Load ground truth image
        gt_img = Image.open(texture_path).convert("RGB")
        self._set_image(self.gt_frame, gt_img)

        thread = threading.Thread(target=self._train_thread, daemon=True)
        thread.start()

    def _train_thread(self):
        texture_path = self.texture_path_var.get()
        hidden = self.hidden_var.get()
        depth = self.depth_var.get()
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()

        self.trainer = TwoInputNeuralBRDFModelTraining(texture_path)
        self.model = TwoInputNeuralBRDFModel(hidden=hidden, depth=depth)

        train_start = time.time()

        def on_epoch(epoch, loss):
            self.losses.append(loss)
            self.after(0, self._update_progress, epoch, epochs, loss)

        losses = self.trainer.train(
            model=self.model,
            lr=lr,
            epochs=epochs,
            batch_size=4096,
            epoch_callback=on_epoch,
        )

        train_elapsed = time.time() - train_start
        epoch_speed = epochs / train_elapsed if train_elapsed > 0 else 0

        export_name = self.export_name_var.get()
        os.makedirs("Assets/Models/Base", exist_ok=True)
        self.model.save(f"Assets/Models/Base/{export_name}.pth")

        infer_start = time.time()
        inferred_img = self.trainer.infer_tensor(self.model)
        infer_elapsed = time.time() - infer_start
        inferred_img.save("Assets/InferredTestTexture.png")

        self.after(
            0,
            self._training_done,
            train_elapsed,
            epoch_speed,
            infer_elapsed,
            inferred_img,
        )

    def _update_progress(self, epoch, total_epochs, loss):
        self.progress_label.config(
            text=f"Epoch {epoch + 1}/{total_epochs}  loss={loss:.6f}"
        )
        self.ax.clear()
        self.ax.set_title("Training Loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.plot(self.losses, color="#4CAF50", linewidth=1.5)
        self.ax.set_xlim(0, max(1, len(self.losses) - 1))
        self.fig.tight_layout()
        self.canvas.draw()

    def _training_done(self, train_elapsed, epoch_speed, infer_elapsed, inferred_img):
        self.training = False
        self.bake_btn.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Done in {train_elapsed:.1f}s. Exporting CoreML...", fg="black"
        )
        self.stat_train_time.config(text=f"{train_elapsed:.2f}s")
        self.stat_epoch_speed.config(text=f"{epoch_speed:.1f} ep/s")
        self.stat_infer_time.config(text=f"{infer_elapsed * 1000:.1f}ms")
        self._set_image(self.inferred_frame, inferred_img)
        self.update()

        # Export CoreML in background so UI stays responsive
        thread = threading.Thread(target=self._export_coreml, daemon=True)
        thread.start()

    def _export_coreml(self):
        export_name = self.export_name_var.get()
        mlpackage_path = f"Assets/Models/Base/{export_name}.mlpackage"
        mtlpackage_path = f"Assets/Models/Metal/{export_name}.mtlpackage"
        try:
            self.model.eval()
            dummy_input = torch.zeros(1, 2)
            traced = torch.jit.trace(self.model, dummy_input)
            coremlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="uv", shape=(1, 2))],
                outputs=[ct.TensorType(name="rgb")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS15,
            )
            coremlmodel.save(mlpackage_path)
            self.after(
                0,
                lambda: self.status_label.config(
                    text="Saved .pth and .mlpackage. Running metal-package-builder...", fg="black"
                ),
            )
        except Exception as e:
            self.after(
                0,
                lambda: self.status_label.config(
                    text=f"CoreML export failed: {e}", fg="red"
                ),
            )
            return

        try:
            os.makedirs("Assets/Models/Metal", exist_ok=True)
            result = subprocess.run(
                ["xcrun", "metal-package-builder", mlpackage_path, "-o", mtlpackage_path],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"Saved .pth, .mlpackage, and .mtlpackage.", fg="green"
                    ),
                )
            else:
                err = result.stderr.strip() or result.stdout.strip()
                self.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"metal-package-builder failed: {err}", fg="red"
                    ),
                )
        except Exception as e:
            self.after(
                0,
                lambda: self.status_label.config(
                    text=f"metal-package-builder error: {e}", fg="red"
                ),
            )


if __name__ == "__main__":
    app = App()
    app.mainloop()
