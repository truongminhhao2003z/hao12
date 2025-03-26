import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  // Khởi tạo state từ localStorage nếu có
  const [tasks, setTasks] = useState(() => {
    const savedTasks = localStorage.getItem('tasks');
    return savedTasks ? JSON.parse(savedTasks) : [];
  });
  const [taskInput, setTaskInput] = useState('');
  const [editingTaskId, setEditingTaskId] = useState(null); // Trạng thái để lưu id task đang chỉnh sửa

  // Lưu danh sách công việc vào localStorage mỗi khi tasks thay đổi
  useEffect(() => {
    localStorage.setItem('tasks', JSON.stringify(tasks));
  }, [tasks]);

  const handleAddTask = () => {
    if (taskInput.trim()) {
      setTasks([...tasks, { id: Date.now(), text: taskInput, completed: false }]);
      setTaskInput('');
    }
  };

  const handleToggleTask = (id) => {
    setTasks(tasks.map(task =>
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  const handleDeleteTask = (id) => {
    setTasks(tasks.filter(task => task.id !== id));
  };

  const handleEditTask = (id, text) => {
    setEditingTaskId(id); // Lưu id task đang được chỉnh sửa
    setTaskInput(text); // Đặt nội dung task vào ô input để sửa
  };

  const handleSaveEdit = () => {
    if (taskInput.trim()) {
      setTasks(tasks.map(task =>
        task.id === editingTaskId ? { ...task, text: taskInput } : task
      ));
      setEditingTaskId(null); // Reset lại id task sau khi chỉnh sửa xong
      setTaskInput('');
    }
  };

  return (
    <div className="App">
      <h1>Nhiệm vụ hằng ngày cho bé</h1>
      <div className="input-container">
        <input
          type="text"
          value={taskInput}
          onChange={(e) => setTaskInput(e.target.value)}
          placeholder="Thêm nhiệm vụ"
        />
        {editingTaskId ? (
          <button className="add-btn" onClick={handleSaveEdit}>Lưu nhiệm vụ</button>
        ) : (
          <button className="add-btn" onClick={handleAddTask}>Thêm</button>
        )}
      </div>
      <ul className="task-list">
        {tasks.map((task) => (
          <li key={task.id} className={task.completed ? 'completed' : ''}>
            <input 
              type="checkbox" 
              checked={task.completed} 
              onChange={() => handleToggleTask(task.id)} 
            />
            <span className="task-text">{task.text}</span>
            <button className="edit-btn" onClick={() => handleEditTask(task.id, task.text)}>Sửa</button>
            <button className="delete-btn" onClick={() => handleDeleteTask(task.id)}>Xóa</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
