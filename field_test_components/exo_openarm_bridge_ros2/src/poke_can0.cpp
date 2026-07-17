#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

using namespace std::chrono_literals;

int main() {
  openarm::can::socket::OpenArm arm("can0", true);

  std::vector<openarm::damiao_motor::MotorType> types = {
    openarm::damiao_motor::MotorType::DM8009,
    openarm::damiao_motor::MotorType::DM8009,
    openarm::damiao_motor::MotorType::DM4340,
    openarm::damiao_motor::MotorType::DM4340,
    openarm::damiao_motor::MotorType::DM4310,
    openarm::damiao_motor::MotorType::DM4310,
    openarm::damiao_motor::MotorType::DM4310
  };

  arm.init_arm_motors(types, {1,2,3,4,5,6,7}, {0x11,0x12,0x13,0x14,0x15,0x16,0x17});
  arm.set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  arm.enable_all();
  std::this_thread::sleep_for(300ms);
  arm.recv_all();

  std::cout << "Poking joint 1 to +0.25 rad for 3 sec\n";

  for (int k = 0; k < 300; ++k) {
    arm.get_arm().mit_control_all({
      {120.0, 2.0, 0.25, 0.0, 0.0},
      {20.0,  1.0, 0.0,  0.0, 0.0},
      {20.0,  1.0, 0.0,  0.0, 0.0},
      {20.0,  1.0, 0.0,  0.0, 0.0},
      {10.0,  0.5, 0.0,  0.0, 0.0},
      {10.0,  0.5, 0.0,  0.0, 0.0},
      {10.0,  0.5, 0.0,  0.0, 0.0}
    });
    arm.recv_all();
    std::this_thread::sleep_for(10ms);
  }

  arm.disable_all();
  arm.recv_all();
  std::cout << "Done, disabled.\n";
}
