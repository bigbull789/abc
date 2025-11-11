// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {
    // Structure to store student details
    struct Student {
        uint256 rollNo;
        string name;
        uint256 marks;
    }

    // Dynamic array to store multiple students
    Student[] public students;

    // Events for logging actions
    event StudentAdded(uint256 rollNo, string name, uint256 marks);
    event Received(address sender, uint256 amount);

    // Function to add a new student
    function addStudent(uint256 _rollNo, string memory _name, uint256 _marks) public {
        students.push(Student(_rollNo, _name, _marks));
        emit StudentAdded(_rollNo, _name, _marks);
    }

    // Function to get total number of students
    function getStudentCount() public view returns (uint256) {
        return students.length;
    }

    //  Function to get student details by Roll Number
    function getStudentByRoll(uint256 _rollNo) public view returns (string memory, uint256) {
        for (uint256 i = 0; i < students.length; i++) {
            if (students[i].rollNo == _rollNo) {
                return (students[i].name, students[i].marks);
            }
        }
        revert("Student not found");
    }

    // Fallback function to handle plain Ether transfers
    fallback() external payable {
        emit Received(msg.sender, msg.value);
    }

    // Receive function (for direct Ether transfer)
    receive() external payable {
        emit Received(msg.sender, msg.value);
    }
}

