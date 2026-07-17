#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

using namespace std::chrono_literals;

static std::atomic<bool> running(true);

void sigint_handler(int) {
  running = false;
}

static double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(x, hi));
}

class ExoToOpenArmCan0 : public rclcpp::Node {
public:
  ExoToOpenArmCan0()
  : Node("exo_to_openarm_can0"),
    arm_("can0", true)
  {
    std::vector<openarm::damiao_motor::MotorType> motor_types = {
      openarm::damiao_motor::MotorType::DM8009,
      openarm::damiao_motor::MotorType::DM8009,
      openarm::damiao_motor::MotorType::DM4340,
      openarm::damiao_motor::MotorType::DM4340,
      openarm::damiao_motor::MotorType::DM4310,
      openarm::damiao_motor::MotorType::DM4310,
      openarm::damiao_motor::MotorType::DM4310
    };

    std::vector<uint32_t> send_ids = {0x01,0x02,0x03,0x04,0x05,0x06,0x07};
    std::vector<uint32_t> recv_ids = {0x11,0x12,0x13,0x14,0x15,0x16,0x17};

    arm_.init_arm_motors(motor_types, send_ids, recv_ids);
    arm_.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);

    arm_.enable_all();
    std::this_thread::sleep_for(100ms);
    arm_.refresh_all();
    arm_.recv_all();

    size_t i = 0;
    for (const auto &m : arm_.get_arm().get_motors()) {
      openarm_zero_[i++] = m.get_position();
    }

    RCLCPP_WARN(this->get_logger(), "can0 motors enabled. Captured OpenArm zero pose.");

    sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/exo/joint_command", 10,
      std::bind(&ExoToOpenArmCan0::exo_cb, this, std::placeholders::_1));

    timer_ = create_wall_timer(20ms, std::bind(&ExoToOpenArmCan0::control_loop, this));
  }

  ~ExoToOpenArmCan0() {
    std::vector<openarm::damiao_motor::MITParam> stop;
    for (int i = 0; i < 7; ++i) stop.push_back({0.0, 0.0, 0.0, 0.0, 0.0});
    arm_.get_arm().mit_control_all(stop);
    arm_.recv_all();
    arm_.disable_all();
    arm_.recv_all();
  }

private:
  void exo_cb(sensor_msgs::msg::JointState::SharedPtr msg) {
    if (msg->position.size() < 7) return;

    std::lock_guard<std::mutex> lock(mtx_);

    if (!exo_zero_set_) {
      for (int i = 0; i < 7; ++i) exo_zero_[i] = msg->position[i];
      exo_zero_set_ = true;
      RCLCPP_WARN(this->get_logger(), "Captured ExoArm zero pose for LEFT arm.");
    }

    for (int i = 0; i < 7; ++i) latest_exo_[i] = msg->position[i];
    got_msg_ = true;
  }

  void control_loop() {
    if (!got_msg_ || !exo_zero_set_) return;

    double target[7];

    {
      std::lock_guard<std::mutex> lock(mtx_);
      for (int i = 0; i < 7; ++i) {
        double delta = scale_ * (latest_exo_[i] - exo_zero_[i]);
        target[i] = openarm_zero_[i] + delta;
      }
    }

    // OpenArm official constants: q command is MITParam.q, order {kp,kd,q,dq,tau}
    std::vector<openarm::damiao_motor::MITParam> cmd;
    for (int i = 0; i < 7; ++i) {
      target[i] = clamp(target[i], -1.0, 1.0);  // first test: very conservative
      cmd.push_back({20.0, 0.5, target[i], 0.0, 0.0});
    }

    arm_.get_arm().mit_control_all(cmd);
    arm_.recv_all();
  }

  openarm::can::socket::OpenArm arm_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mtx_;
  bool got_msg_{false};
  bool exo_zero_set_{false};

  double scale_{0.2};
  double exo_zero_[7]{};
  double latest_exo_[7]{};
  double openarm_zero_[7]{};
};

int main(int argc, char **argv) {
  std::signal(SIGINT, sigint_handler);
  rclcpp::init(argc, argv);

  auto node = std::make_shared<ExoToOpenArmCan0>();

  while (rclcpp::ok() && running) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(1ms);
  }

  node.reset();
  rclcpp::shutdown();
  return 0;
}
