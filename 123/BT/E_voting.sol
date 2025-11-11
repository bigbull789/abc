// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract EVoting {
    address public owner;
    bool public electionActive;
    uint public candidateCount;

    struct Candidate {
        uint id;
        string name;
        uint votes;
    }

    struct Voter {
        bool voted;
    }

    mapping(uint => Candidate) public candidates;
    mapping(address => Voter) public voters;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner allowed");
        _;
    }

    function addCandidate(string memory _name) public onlyOwner {
        require(!electionActive, "Election started");
        candidateCount++;
        candidates[candidateCount] = Candidate(candidateCount, _name, 0);
    }

    function startElection() public onlyOwner {
        require(!electionActive, "Already active");
        electionActive = true;
    }

    function endElection() public onlyOwner {
        require(electionActive, "Not active");
        electionActive = false;
    }

    function vote(uint _id) public {
        require(electionActive, "Election not active");
        require(!voters[msg.sender].voted, "Already voted");
        require(_id > 0 && _id <= candidateCount, "Invalid ID");

        voters[msg.sender].voted = true;
        candidates[_id].votes++;
    }

    function getCandidate(uint _id) public view returns (string memory, uint) {
        Candidate storage c = candidates[_id];
        return (c.name, c.votes);
    }
}
