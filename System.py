#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chương trình nhận diện âm thanh trực tiếp và chuyển đổi sang ngôn ngữ ký hiệu
Hỗ trợ người khiếm thính - Phiên bản cải tiến
"""

import speech_recognition as sr
import pyaudio
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from datetime import datetime
import json
import sys

class AudioRecorder:
    """Class để ghi âm và xử lý audio"""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recognizer = sr.Recognizer()
        
        # Tối ưu hóa recognizer
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
        
        print("🎤 Đang khởi tạo microphone...")
        self.setup_microphone()
        
    def setup_microphone(self):
        """Thiết lập và kiểm tra microphone"""
        try:
            # Liệt kê các thiết bị âm thanh có sẵn
            print("📋 Danh sách microphone:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"   {index}: {name}")
            
            # Sử dụng microphone mặc định
            self.microphone = sr.Microphone()
            
            # Hiệu chỉnh nhiễu môi trường
            with self.microphone as source:
                print("⚙️ Đang hiệu chỉnh nhiễu môi trường...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"✅ Ngưỡng năng lượng: {self.recognizer.energy_threshold}")
                
        except Exception as e:
            print(f"❌ Lỗi thiết lập microphone: {e}")
            raise
            
    def start_recording(self, callback):
        """Bắt đầu ghi âm"""
        if self.is_recording:
            return False
            
        self.is_recording = True
        self.callback = callback
        
        # Thread ghi âm
        self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.record_thread.start()
        
        # Thread xử lý audio
        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.process_thread.start()
        
        print("🎙️ Bắt đầu ghi âm...")
        return True
        
    def stop_recording(self):
        """Dừng ghi âm"""
        if not self.is_recording:
            return False
            
        self.is_recording = False
        print("⏹️ Đã dừng ghi âm")
        return True
        
    def _record_audio(self):
        """Thread ghi âm liên tục"""
        while self.is_recording:
            try:
                with self.microphone as source:
                    print("👂 Đang lắng nghe... (nói gì đó)")
                    # Ghi âm với timeout ngắn để responsive hơn
                    audio_data = self.recognizer.listen(
                        source, 
                        timeout=1,
                        phrase_time_limit=5
                    )
                    
                    if audio_data:
                        self.audio_queue.put(audio_data)
                        print(f"📼 Đã ghi được {len(audio_data.frame_data)} bytes")
                        
            except sr.WaitTimeoutError:
                # Timeout bình thường, tiếp tục
                continue
            except Exception as e:
                print(f"❌ Lỗi ghi âm: {e}")
                time.sleep(0.5)
                
    def _process_audio(self):
        """Thread xử lý audio thành text"""
        while self.is_recording:
            try:
                # Lấy audio từ queue
                audio_data = self.audio_queue.get(timeout=1)
                print("🔄 Đang xử lý audio...")
                
                # Thử nhiều engine khác nhau
                text = self._recognize_audio(audio_data)
                
                if text and text.strip():
                    print(f"✅ Nhận diện được: {text}")
                    self.callback(text.strip())
                else:
                    print("🔇 Không nhận diện được từ nào")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Lỗi xử lý audio: {e}")
                
    def _recognize_audio(self, audio_data):
        """Nhận diện audio thành text với nhiều phương pháp"""
        methods = [
            ("Google (vi-VN)", lambda: self.recognizer.recognize_google(audio_data, language='vi-VN')),
            ("Google (en-US)", lambda: self.recognizer.recognize_google(audio_data, language='en-US')),
            ("Sphinx", lambda: self.recognizer.recognize_sphinx(audio_data)),
        ]
        
        for method_name, recognize_func in methods:
            try:
                print(f"🔍 Thử {method_name}...")
                result = recognize_func()
                if result and result.strip():
                    print(f"✅ {method_name} thành công: {result}")
                    return result
            except sr.UnknownValueError:
                print(f"🔇 {method_name}: Không nhận diện được")
                continue
            except sr.RequestError as e:
                print(f"❌ {method_name}: Lỗi dịch vụ - {e}")
                continue
            except Exception as e:
                print(f"❌ {method_name}: Lỗi - {e}")
                continue
                
        return None


class SignLanguageConverter:
    """Class chuyển đổi text sang ký hiệu"""
    def __init__(self):
        self.sign_dictionary = {
            # Chào hỏi cơ bản
            'xin chào': '👋 Vẫy tay chào',
            'chào': '👋 Vẫy tay',
            'hello': '👋 Wave hand',
            'hi': '👋 Wave hand',
            'cảm ơn': '🙏 Chắp tay cảm ơn',
            'thank you': '🙏 Thank you gesture',
            'thanks': '🙏 Thank you gesture',
            
            # Đại từ
            'tôi': '👤 Chỉ vào bản thân',
            'i': '👤 Point to self',
            'bạn': '👥 Chỉ về phía người khác',
            'you': '👥 Point to other person',
            'chúng tôi': '👫 Chỉ nhóm người',
            'we': '👫 Point to group',
            
            # Gia đình
            'gia đình': '👨‍👩‍👧‍👦 Vòng tròn với tay',
            'family': '👨‍👩‍👧‍👦 Family circle',
            'mẹ': '👩 Tay chạm vào cằm',
            'mother': '👩 Touch chin',
            'bố': '👨 Tay chạm vào trán',
            'father': '👨 Touch forehead',
            'con': '👶 Tay ru em bé',
            'child': '👶 Rock baby',
            
            # Cảm xúc
            'yêu': '❤️ Tay tạo hình trái tim',
            'love': '❤️ Heart shape with hands',
            'vui': '😊 Tay kéo miệng cười',
            'happy': '😊 Pull mouth to smile',
            'buồn': '😢 Tay vuốt nước mắt',
            'sad': '😢 Wipe tears',
            'tức giận': '😠 Cau mày tức giận',
            'angry': '😠 Angry expression',
            
            # Hoạt động
            'ăn': '🍽️ Tay đưa về miệng',
            'eat': '🍽️ Bring hand to mouth',
            'uống': '🥤 Tay nghiêng về miệng',
            'drink': '🥤 Tilt hand to mouth',
            'ngủ': '😴 Nghiêng đầu nằm xuống',
            'sleep': '😴 Tilt head down',
            'đi': '🚶 Hai ngón tay bước đi',
            'go': '🚶 Two fingers walking',
            'về': '🔙 Tay chỉ về phía sau',
            'come': '🔙 Point backwards',
            'làm việc': '💼 Tay gõ máy tính',
            'work': '💼 Typing motion',
            'học': '📚 Mở sách đọc',
            'study': '📚 Open book',
            
            # Từ phổ biến
            'tốt': '👍 Giơ ngón cái lên',
            'good': '👍 Thumbs up',
            'xấu': '👎 Ngón cái xuống',
            'bad': '👎 Thumbs down',
            'có': '✅ Gật đầu đồng ý',
            'yes': '✅ Nod head',
            'không': '❌ Lắc đầu từ chối',
            'no': '❌ Shake head',
            'lớn': '📏 Tay duỗi ra xa',
            'big': '📏 Stretch arms wide',
            'nhỏ': '🤏 Tay chụm lại',
            'small': '🤏 Pinch fingers',
            
            # Màu sắc
            'đỏ': '🔴 Chỉ vào môi',
            'red': '🔴 Point to lips',
            'xanh': '🔵 Chỉ lên trời',
            'blue': '🔵 Point to sky',
            'vàng': '🟡 Chỉ vào mặt trời',
            'yellow': '🟡 Point to sun',
            
            # Số đếm
            'một': '☝️ Một ngón tay',
            'one': '☝️ One finger',
            'hai': '✌️ Hai ngón tay',
            'two': '✌️ Two fingers',
            'ba': '👌 Ba ngón tay',
            'three': '👌 Three fingers',
        }
        
    def convert_text_to_signs(self, text):
        """Chuyển đổi text thành ký hiệu"""
        if not text:
            return []
            
        # Làm sạch text
        text = text.lower().strip()
        words = text.replace(',', ' ').replace('.', ' ').split()
        
        results = []
        for word in words:
            if not word:
                continue
                
            # Tìm kiếm chính xác
            if word in self.sign_dictionary:
                results.append((word, self.sign_dictionary[word], "exact"))
                continue
                
            # Tìm kiếm từng phần
            found = False
            for key, value in self.sign_dictionary.items():
                if word in key or key in word:
                    results.append((word, value, "partial"))
                    found = True
                    break
                    
            if not found:
                results.append((word, f"❓ Chưa có ký hiệu cho '{word}'", "missing"))
                
        return results


class SpeechToSignGUI:
    """Giao diện người dùng"""
    def __init__(self):
        self.recorder = AudioRecorder()
        self.converter = SignLanguageConverter()
        self.is_listening = False
        
        self.setup_gui()
        self.test_components()
        
    def test_components(self):
        """Test các thành phần"""
        print("🧪 Đang test các thành phần...")
        
        # Test converter
        test_text = "xin chào tôi yêu bạn"
        signs = self.converter.convert_text_to_signs(test_text)
        print(f"✅ Converter test: '{test_text}' -> {len(signs)} ký hiệu")
        
        # Test microphone
        try:
            with self.recorder.microphone as source:
                self.recorder.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("✅ Microphone test: OK")
        except Exception as e:
            print(f"❌ Microphone test: {e}")
        
    def setup_gui(self):
        """Thiết lập giao diện"""
        self.root = tk.Tk()
        self.root.title("🤟 Nhận diện Giọng nói sang Ký hiệu - Phiên bản cải tiến")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2c3e50')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        self.setup_header()
        
        # Control panel
        self.setup_controls()
        
        # Status
        self.setup_status()
        
        # Content area
        self.setup_content()
        
        # Footer
        self.setup_footer()
        
    def setup_header(self):
        """Thiết lập header"""
        header_frame = tk.Frame(self.root, bg='#34495e', height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="🤟 Nhận diện Giọng nói sang Ngôn ngữ Ký hiệu",
            font=('Arial', 20, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        title.pack(expand=True)
        
        subtitle = tk.Label(
            header_frame,
            text="Phiên bản cải tiến - Hỗ trợ đa ngôn ngữ",
            font=('Arial', 11),
            bg='#34495e',
            fg='#bdc3c7'
        )
        subtitle.pack()
        
    def setup_controls(self):
        """Thiết lập điều khiển"""
        control_frame = tk.Frame(self.root, bg='#2c3e50', pady=20)
        control_frame.pack()
        
        # Buttons
        self.start_btn = tk.Button(
            control_frame,
            text="🎤 Bắt đầu nghe",
            command=self.toggle_listening,
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=30,
            pady=15,
            relief='raised',
            cursor='hand2'
        )
        self.start_btn.pack(side='left', padx=10)
        
        self.clear_btn = tk.Button(
            control_frame,
            text="🗑️ Xóa tất cả",
            command=self.clear_all,
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            padx=30,
            pady=15,
            relief='raised',
            cursor='hand2'
        )
        self.clear_btn.pack(side='left', padx=10)
        
        self.test_btn = tk.Button(
            control_frame,
            text="🧪 Test",
            command=self.test_recognition,
            font=('Arial', 14, 'bold'),
            bg='#9b59b6',
            fg='white',
            padx=30,
            pady=15,
            relief='raised',
            cursor='hand2'
        )
        self.test_btn.pack(side='left', padx=10)
        
        self.save_btn = tk.Button(
            control_frame,
            text="💾 Lưu",
            command=self.save_results,
            font=('Arial', 14, 'bold'),
            bg='#e67e22',
            fg='white',
            padx=30,
            pady=15,
            relief='raised',
            cursor='hand2'
        )
        self.save_btn.pack(side='left', padx=10)
        
    def setup_status(self):
        """Thiết lập thanh trạng thái"""
        self.status_var = tk.StringVar(value="🎤 Sẵn sàng - Nhấn 'Bắt đầu nghe' để bắt đầu")
        
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='#2c3e50',
            pady=10
        )
        self.status_label.pack(fill='x')
        
    def setup_content(self):
        """Thiết lập nội dung chính"""
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Text input (for testing)
        input_frame = tk.LabelFrame(
            main_frame,
            text="📝 Nhập text để test (không bắt buộc)",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        input_frame.pack(fill='x', pady=(0, 10))
        
        input_container = tk.Frame(input_frame)
        input_container.pack(fill='x', padx=10, pady=10)
        
        self.text_input = tk.Entry(
            input_container,
            font=('Arial', 12),
            bg='white'
        )
        self.text_input.pack(side='left', fill='x', expand=True)
        
        test_text_btn = tk.Button(
            input_container,
            text="➤ Test",
            command=self.test_text_input,
            font=('Arial', 10),
            bg='#3498db',
            fg='white',
            padx=15
        )
        test_text_btn.pack(side='right', padx=(10, 0))
        
        # Recognized text
        text_frame = tk.LabelFrame(
            main_frame,
            text="🎙️ Văn bản được nhận diện",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        text_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.text_display = scrolledtext.ScrolledText(
            text_frame,
            font=('Arial', 11),
            height=10,
            wrap='word',
            bg='white',
            fg='#2c3e50'
        )
        self.text_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Sign language output
        sign_frame = tk.LabelFrame(
            main_frame,
            text="🤲 Ngôn ngữ ký hiệu tương ứng",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        sign_frame.pack(fill='both', expand=True)
        
        self.sign_display = scrolledtext.ScrolledText(
            sign_frame,
            font=('Arial', 11),
            height=10,
            wrap='word',
            bg='#f8f9fa',
            fg='#2c3e50'
        )
        self.sign_display.pack(fill='both', expand=True, padx=10, pady=10)
        
    def setup_footer(self):
        """Thiết lập footer"""
        self.stats_var = tk.StringVar(
            value=f"📊 Từ điển: {len(self.converter.sign_dictionary)} từ | Đã xử lý: 0 câu"
        )
        
        footer = tk.Label(
            self.root,
            textvariable=self.stats_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='#bdc3c7',
            pady=5
        )
        footer.pack(fill='x')
        
        self.processed_sentences = 0
        
    def toggle_listening(self):
        """Bật/tắt lắng nghe"""
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()
            
    def start_listening(self):
        """Bắt đầu lắng nghe"""
        try:
            if self.recorder.start_recording(self.on_speech_recognized):
                self.is_listening = True
                self.start_btn.configure(
                    text="⏹️ Dừng nghe",
                    bg='#e74c3c'
                )
                self.status_var.set("🎧 Đang lắng nghe... Hãy nói gì đó!")
                print("✅ Bắt đầu lắng nghe thành công")
            else:
                messagebox.showerror("Lỗi", "Không thể bắt đầu lắng nghe")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi bắt đầu lắng nghe: {e}")
            
    def stop_listening(self):
        """Dừng lắng nghe"""
        if self.recorder.stop_recording():
            self.is_listening = False
            self.start_btn.configure(
                text="🎤 Bắt đầu nghe",
                bg='#27ae60'
            )
            self.status_var.set("🎤 Đã dừng lắng nghe")
            
    def on_speech_recognized(self, text):
        """Callback khi nhận diện được giọng nói"""
        # Sử dụng after để update GUI từ thread khác
        self.root.after(0, self._update_gui_with_text, text)
        
    def _update_gui_with_text(self, text):
        """Cập nhật GUI với text nhận diện được"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Hiển thị text được nhận diện
        self.text_display.insert('end', f"[{timestamp}] {text}\n")
        self.text_display.see('end')
        
        # Chuyển đổi sang ký hiệu
        signs = self.converter.convert_text_to_signs(text)
        
        # Hiển thị ký hiệu
        self.sign_display.insert('end', f"\n--- [{timestamp}] ---\n")
        
        if signs:
            for word, sign, match_type in signs:
                color_indicator = {
                    'exact': '✅',
                    'partial': '🟡', 
                    'missing': '❌'
                }.get(match_type, '❓')
                
                self.sign_display.insert('end', f"{color_indicator} '{word}' → {sign}\n")
        else:
            self.sign_display.insert('end', "❓ Không tìm thấy ký hiệu phù hợp\n")
            
        self.sign_display.insert('end', "\n")
        self.sign_display.see('end')
        
        # Cập nhật thống kê
        self.processed_sentences += 1
        self.stats_var.set(
            f"📊 Từ điển: {len(self.converter.sign_dictionary)} từ | "
            f"Đã xử lý: {self.processed_sentences} câu"
        )
        
        # Cập nhật status
        preview = text[:30] + "..." if len(text) > 30 else text
        self.status_var.set(f"✅ Vừa xử lý: '{preview}'")
        
    def test_text_input(self):
        """Test với text nhập vào"""
        text = self.text_input.get().strip()
        if text:
            self._update_gui_with_text(text)
            self.text_input.delete(0, 'end')
        
    def test_recognition(self):
        """Test với câu mẫu"""
        test_sentences = [
            "xin chào tôi yêu bạn",
            "cảm ơn gia đình tôi",
            "hello i love you",
            "tôi đi học về nhà ăn cơm",
            "mẹ bố yêu con"
        ]
        
        import random
        test_text = random.choice(test_sentences)
        self._update_gui_with_text(test_text)
        
    def clear_all(self):
        """Xóa tất cả"""
        self.text_display.delete('1.0', 'end')
        self.sign_display.delete('1.0', 'end')
        self.text_input.delete(0, 'end')
        self.processed_sentences = 0
        self.stats_var.set(
            f"📊 Từ điển: {len(self.converter.sign_dictionary)} từ | Đã xử lý: 0 câu"
        )
        self.status_var.set("🗑️ Đã xóa tất cả nội dung")
        
    def save_results(self):
        """Lưu kết quả"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Lưu kết quả nhận diện"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== KẾT QUẢ NHẬN DIỆN GIỌNG NÓI SANG KÝ HIỆU ===\n\n")
                    f.write("📝 VĂN BẢN NHẬN DIỆN:\n")
                    f.write(self.text_display.get('1.0', 'end'))
                    f.write("\n🤲 NGÔN NGỮ KÝ HIỆU:\n")
                    f.write(self.sign_display.get('1.0', 'end'))
                    f.write(f"\n📊 ĐÃ XỬ LÝ: {self.processed_sentences} câu\n")
                    f.write(f"📚 TỪ ĐIỂN: {len(self.converter.sign_dictionary)} từ")
                    
                messagebox.showinfo("Thành công", f"Đã lưu vào:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu: {e}")
            
    def run(self):
        """Chạy ứng dụng"""
        try:
            print("🚀 Khởi chạy giao diện...")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"❌ Lỗi chạy ứng dụng: {e}")
            
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        if self.is_listening:
            self.stop_listening()
        self.root.destroy()


def check_requirements():
    """Kiểm tra thư viện cần thiết"""
    required_packages = {
        'speech_recognition': 'SpeechRecognition',
        'pyaudio': 'pyaudio',
        'tkinter': 'tkinter (built-in)',
    }
    
    missing = []
    for package, install_name in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package}: OK")
        except ImportError:
            missing.append(install_name)
            print(f"❌ {package}: Thiếu")
    
    if missing:
        print("\n📦 Cài đặt thư viện thiếu:")
        for package in missing:
            if package != 'tkinter (built-in)':
                print(f"   pip install {package}")
        return False
    return True


def main():
    """Hàm main"""
    print("=" * 60)
    print("🤟 CHƯƠNG TRÌNH NHẬN DIỆN GIỌNG NÓI SANG KÝ HIỆU")
    print("=" * 60)
    
    # Kiểm tra thư viện
    if not check_requirements():
        print("\n❌ Vui lòng cài đặt thư viện thiếu trước khi chạy!")
        return
    
    try:
        print("🚀 Khởi tạo ứng dụng...")
        app = SpeechToSignGUI()
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Đã thoát!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()